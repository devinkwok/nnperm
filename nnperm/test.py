import unittest
from copy import deepcopy
from itertools import product
from collections import defaultdict
import torch
import numpy as np
import torch
from torch import nn

import sys
sys.path.append("open_lth")
from open_lth.models import cifar_resnet
from open_lth.models import cifar_vgg
from open_lth.models.initializers import kaiming_normal

from nnperm.spec import PermutationSpec, ScaleSpec
from nnperm.align import *
from nnperm.eval import evaluate_model
from nnperm.barrier import EnsembleModel, interpolate_dict
from nnperm.utils import to_torch_device, keys_match, multiplicative_weight_noise, to_numpy


class TestNNPerm(unittest.TestCase):

    def _make_dataloader(self, tensor):
        dataset = torch.utils.data.TensorDataset(tensor, torch.zeros(tensor.shape[0], dtype=torch.long))
        return torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    def setUp(self) -> None:
        ## Setup
        self.models = [
            (
                nn.Sequential(
                    nn.Linear(20, 10),
                    nn.ReLU(),
                    nn.Linear(10, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2),
                ),
                self._make_dataloader(torch.randn([16, 20])),
            ),
            (
                cifar_vgg.Model.get_model_from_name("cifar_vgg_11", initializer=kaiming_normal),
                self._make_dataloader(torch.randn([16, 3, 32, 32])),
            ),
            # (
            #     cifar_resnet.Model.get_model_from_name(
            #           "cifar_resnet_20_4", initializer=kaiming_normal),
            #     self._make_dataloader(torch.randn([16, 3, 32, 32])),
            # ),
        ]
        for model, _ in self.models:
            self._randomize_batchnorm_weights(model)
            self._insert_small_weights(model, 0.05)

    def _randomize_batchnorm_weights(self, model):
        state_dict = model.state_dict()
        with torch.no_grad():
            for k, v in state_dict.items():
                if "running_mean" in k:
                    key = k[:-len(".running_mean")]
                    tensors = [(k, v),
                        ("weight", state_dict[key + ".weight"]),
                        ("bias", state_dict[key + ".bias"]),
                        ("running_mean", state_dict[key + ".running_mean"]),
                        ("running_var", state_dict[key + ".running_var"]),
                    ]
                    for _, t in tensors:
                        assert len(v.shape) == 1 and v.shape == t.shape, (k, v.shape, t.shape)
                        t.add_(torch.rand_like(t) - 0.5)

    def _insert_small_weights(self, model, probability, scale_factor=1e-2):
        state_dict = model.state_dict()
        for k, v in state_dict.items():
            if "weight" in k:
                mask = torch.rand_like(v) < probability
                scale = torch.ones_like(v)
                scale[mask] = scale_factor
                state_dict[k] *= scale

    class StateUnchangedContextManager():
        def __init__(self, state) -> None:
            self.state = state
            self.original = deepcopy(state)

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_value, exc_tb):
            assert np.all([k in self.original for k in self.state.keys()])
            assert np.all([k in self.state for k in self.original.keys()])
            for k, v in self.state.items():
                if isinstance(v, torch.Tensor):
                    assert torch.equal(self.original[k], v)
                else:
                    assert np.all(self.original[k] == v)

    def _validate_symmetry(self, model, data, transform_fn):
        output = evaluate_model(model, data, device="cpu")
        state_dict = model.state_dict()
        with self.StateUnchangedContextManager(state_dict):
            transformed = to_torch_device(transform_fn(state_dict), "cpu")
            transformed_output = evaluate_model(model, data, state_dict=transformed, device="cpu")
            np.testing.assert_allclose(output, transformed_output, rtol=1e-4, atol=1e-4)

    def _validate_perm_align(self, model, data, perm_spec, perm, sklearn_like_obj, noise_std=0., allowed_errors=0, allowed_loss=1e-4, tries=2):
        state_dict = model.state_dict()
        # repeat to make sure align_obj isn't reusing old params
        for i in range(tries):
            with self.StateUnchangedContextManager(state_dict):
                permuted_state_dict = perm_spec.apply_permutation(state_dict, perm)
                permuted_state_dict = multiplicative_weight_noise(permuted_state_dict, noise_std)
                found_perm, *_ = sklearn_like_obj.fit(permuted_state_dict, state_dict)
                transform_fn = lambda x: sklearn_like_obj.fit_transform(permuted_state_dict, x)[0]
                self._validate_symmetry(model, data, transform_fn)
            self.assertTrue(keys_match(perm, found_perm))
            for k, v in perm.items():
                self.assertLessEqual(np.count_nonzero(v != found_perm[k]), allowed_errors)

    def _get_perm_spec(self, state_dict):
        if any("block" in k for k in state_dict.keys()):
            return PermutationSpec.from_residual_model(state_dict)
        return PermutationSpec.from_sequential_model(state_dict)

    def test_apply(self):
        for model, _ in self.models:
            state_dict = model.state_dict()
            perm_spec = self._get_perm_spec(state_dict)
            sizes = perm_spec.get_sizes(state_dict)
            self.assertEqual(len(sizes), len(perm_spec.group_to_axes))
            padded = perm_spec.apply_padding(state_dict, {k: v + 5 for k, v in sizes.items()})
            self.assertTrue(keys_match(state_dict, padded))
            original_shapes = np.concatenate([v.shape for k, v in state_dict.items()])
            padded_shapes = np.concatenate([v.shape for k, v in padded.items()])
            self.assertTrue(np.all(np.logical_or(
                padded_shapes == original_shapes, padded_shapes == original_shapes + 5)))
            perms = perm_spec.get_random_permutation(state_dict)
            self.assertTrue(keys_match(sizes, perms))
            for k, v in sizes.items():
                self.assertEqual(v, len(perms[k]))

    def test_perm(self):
        for model, data in self.models:
            with self.subTest(model=model):
                state_dict = model.state_dict()
                perm_spec = self._get_perm_spec(state_dict)
                transform_fn = lambda x: perm_spec.apply_rand_perm(x)
                self._validate_symmetry(model, data, transform_fn)
                rand_perm = perm_spec.get_random_permutation(state_dict)
                new_state_dict = perm_spec.apply_permutation(perm_spec.apply_permutation(
                    state_dict, rand_perm), rand_perm.inverse())
                identity = rand_perm.compose(rand_perm.inverse()).to_matrices()
                for k in state_dict.keys():
                    np.testing.assert_array_equal(state_dict[k], new_state_dict[k])
                for x in identity.values():
                    np.testing.assert_array_equal(x, np.eye(len(x)))

    def test_subset_perm(self):
        for model, data in self.models:
            with self.subTest(model=model):
                state_dict = model.state_dict()
                perm_spec = self._get_perm_spec(state_dict)
                ps2 = perm_spec.subset()
                self.assertDictEqual(perm_spec.axes_to_group, ps2.axes_to_group)
                self.assertDictEqual(perm_spec.group_to_axes, ps2.group_to_axes)

                one_perm = list(perm_spec.group_to_axes.keys())[0:1]
                ps2 = perm_spec.subset(include_groups=one_perm)
                self.assertListEqual(list(ps2.group_to_axes.keys()), one_perm)
                for v in ps2.axes_to_group.values():
                    for k in v:
                        self.assertTrue(k is None or k[0] == one_perm[0])

                one_layer = list(perm_spec.axes_to_group.keys())[0:1]
                ps2 = perm_spec.subset(include_axes=one_layer)
                self.assertListEqual(list(ps2.axes_to_group.keys()), one_layer)
                for v in ps2.group_to_axes.values():
                    for k, _, _ in v:
                        self.assertEqual(k, one_layer[0])

    def test_fit_transform(self):
        for model, _ in self.models:
            state_dict = model.state_dict()
            perm_spec = self._get_perm_spec(state_dict)
            target_params = perm_spec.apply_rand_perm(state_dict)
            perm_params = []
            with self.StateUnchangedContextManager(target_params):
                align_obj = WeightAlignment(perm_spec, kernel="linear")
                perm_params.append(align_obj.fit_transform(target_params, state_dict)[0])
                align_obj.fit(target_params, state_dict)
                perm_params.append(align_obj.transform()[0])
                perm_params.append(align_obj.fit_transform(target_params, state_dict)[0])
                align_obj.fit(target_params, state_dict)
                perm_params.append(align_obj.transform()[0])
                perm_params.append(align_obj.params_)
            for param in perm_params:
                for k, v in target_params.items():
                    np.testing.assert_array_equal(v, param[k])

    def test_align(self):
        for seed in [100, 200]:
            self.setUp()
            for model, data in self.models:
                state_dict = model.state_dict()
                perm_spec = self._get_perm_spec(state_dict)
                for weight_noise, kernel, order, make_perm_fn in product(
                    [0., 1e-1], ["linear", "cosine", "loglinear"], ["forward", "backward", "random"],
                    [perm_spec.get_random_permutation],
                ):
                    with self.subTest(model=model, weight_noise=weight_noise, kernel=kernel, order=order, perm_fn=make_perm_fn):
                        align_obj = WeightAlignment(perm_spec, kernel=kernel, seed=seed, order=order)
                        perm = make_perm_fn(state_dict)
                        self._validate_perm_align(model, data, perm_spec, perm, align_obj, noise_std=weight_noise)

    def test_partial_align(self):
        # can align when ignoring layers that differ in size
        params_a = self.models[0][0].state_dict()
        params_b = {k: v for k, v in params_a.items()}
        weight_k = list(params_a.keys())[-2]
        bias_k = list(params_a.keys())[-1]
        params_b[bias_k] = np.ones(17)
        wider_shape = list(params_a[weight_k].shape)
        wider_shape[0] = 17
        params_b[weight_k] = np.ones(wider_shape)
        perm_spec = self._get_perm_spec(params_a)
        # should fail with original spec since the last layers differ in size
        align_obj = WeightAlignment(perm_spec, kernel="linear")
        self.assertRaises(ValueError, lambda: align_obj.fit(params_a, params_b))
        # should succeed if the differing size layers are removed from perm_spec
        subset_spec = perm_spec.subset(exclude_axes=[weight_k, bias_k])
        align_obj = WeightAlignment(subset_spec, kernel="linear")
        align_obj.fit(perm_spec.apply_rand_perm(params_a), params_b)
        mask = {k: np.random.randn(*v.shape) for k, v in params_a.items()}
        permuted_mask, _ = align_obj.transform(mask)
        for k in mask.keys():
            if k == weight_k or k == bias_k:  # excluded layers should not be permuted
                np.testing.assert_array_equal(mask[k], permuted_mask[k])
            else:  # all other layers are permuted
                self.assertTrue(np.any(mask[k] != permuted_mask[k]))

    def test_activation_align(self):
        for model, data in self.models:
            state_dict = model.state_dict()
            perm_spec = self._get_perm_spec(state_dict)
            for weight_noise, kernel, intermediate_type, make_perm_fn in product(
                [0., 1e-8], ["linear", "cosine"], ["first", "all", "last"], [perm_spec.get_random_permutation],
            ):
                with self.subTest(model=model, weight_noise=weight_noise, intermediate_type=intermediate_type, kernel=kernel, perm_fn=make_perm_fn):
                    align_obj = ActivationAlignment(perm_spec, data, model, kernel=kernel, device="cpu")
                    perm = make_perm_fn(state_dict)
                    self._validate_perm_align(model, data, perm_spec, perm, align_obj, noise_std=weight_noise)

    def test_bootstrap(self):
        for model, data in self.models:
            state_dict = model.state_dict()
            perm_spec = self._get_perm_spec(state_dict)
            for kernel, order, make_perm_fn in product(
                ["linear_bootstrap_1000"], ["random"],
                [ perm_spec.get_random_permutation],
            ):
                with self.subTest(model=model, kernel=kernel, order=order, perm_fn=make_perm_fn):
                    align_obj = WeightAlignment(perm_spec, kernel=kernel, order=order)
                    perm = make_perm_fn(state_dict)
                    self._validate_perm_align(model, data, perm_spec, perm, align_obj,
                                            allowed_errors=10, allowed_loss=1e18)

    class TestModel(nn.Module):
        def __init__(self, n_in, n_h1, n_h2, n_out):
            super().__init__()
            self.linear1 = nn.Linear(n_in, n_h1)
            self.linear2 = nn.Linear(n_h1, n_h2)
            self.linear3 = nn.Linear(n_h2, n_out)

        def forward(self, x):
            x = nn.functional.relu(self.linear1(x))
            x = nn.functional.relu(self.linear2(x))
            x = self.linear3(x)
            return x

        def debug_forward(self, x):
            x = nn.functional.relu(self.linear1(x))
            x = nn.functional.relu(self.linear2(x))
            # x = self.linear3(x)
            return x

    def _make_conjoined_network(self, n_in, n_out, h1, h2, clone=[]):
        full_model = self.TestModel(n_in, h1*2, h2*2, n_out)
        # make hidden layer weights block diagonal to get 2 independent sub-models
        full_model.linear2.weight[:h2, h1:] = 0.
        full_model.linear2.weight[h2:, :h1] = 0.
        if 1 in clone:
            full_model.linear1.weight[h1:, :] = full_model.linear1.weight[:h1, :]
            full_model.linear1.bias[h1:] = full_model.linear1.bias[:h1]
        if 2 in clone:
            full_model.linear2.weight[:h2, :h1] = full_model.linear2.weight[h2:, h1:]
            full_model.linear2.bias[h2:] = full_model.linear2.bias[:h2]
        if 3 in clone:
            full_model.linear3.weight[:, h2:] = full_model.linear3.weight[:, :h2]
        model_a = self.TestModel(n_in, h1, h2, n_out)
        model_b = self.TestModel(n_in, h1, h2, n_out)
        model_a.linear1.weight = nn.Parameter(full_model.linear1.weight[:h1, :])
        model_b.linear1.weight = nn.Parameter(full_model.linear1.weight[h1:, :])
        model_a.linear1.bias = nn.Parameter(full_model.linear1.bias[:h1])
        model_b.linear1.bias = nn.Parameter(full_model.linear1.bias[h1:])
        model_a.linear2.weight = nn.Parameter(full_model.linear2.weight[:h2, :h1])
        model_b.linear2.weight = nn.Parameter(full_model.linear2.weight[h2:, h1:])
        model_a.linear2.bias = nn.Parameter(full_model.linear2.bias[:h2])
        model_b.linear2.bias = nn.Parameter(full_model.linear2.bias[h2:])
        model_a.linear3.weight = nn.Parameter(full_model.linear3.weight[:, :h2])
        model_b.linear3.weight = nn.Parameter(full_model.linear3.weight[:, h2:])
        # biases are divided by 2 because they are added twice in ensemble model
        model_a.linear3.bias = nn.Parameter(full_model.linear3.bias / 2)
        model_b.linear3.bias = nn.Parameter(full_model.linear3.bias / 2)
        align_mask = {k: 0. for k in full_model.state_dict().keys()}
        return full_model, model_a, model_b, align_mask

    def test_layerwise_ensembling(self):
        x = next(iter(self.models[0][1]))[0]
        with torch.no_grad():
            for clone in [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]:
                full_model, model_a, model_b, align_mask = self._make_conjoined_network(20, 2, 6, 3, clone)
                ensemble_model = EnsembleModel(model_a, model_a.state_dict(), model_b.state_dict())
                self.assertTrue(torch.allclose(ensemble_model(x) * 2, full_model(x)))

    def _state_dicts_equal(self, *models, ignore_keys=[]):
        state_dict = models[0]
        if not isinstance(state_dict, dict):
            state_dict = state_dict.state_dict()
        key = next(iter(state_dict.keys()))
        for i, other_dict in enumerate(models[1:]):
            if not isinstance(other_dict, dict):
                other_dict = other_dict.state_dict()
            for k in state_dict.keys():
                if k not in ignore_keys:
                    self.assertTrue(torch.all(state_dict[k] == other_dict[k]), f"Model 0 not equal with {i}")

    def test_ensemble_model_interpolation(self):
        model_a = self.TestModel(20, 6, 3, 2)
        model_b = self.TestModel(20, 6, 3, 2)
        params_a = model_a.state_dict()
        params_b = model_b.state_dict()
        first_layer_aligned = defaultdict(lambda: 0.)
        first_layer_aligned["linear1.weight"] = 1.
        ensemble_a, ensemble_b = EnsembleModel.models_for_interpolation(model_a, params_a, params_b, first_layer_aligned)
        self._state_dicts_equal(model_a, ensemble_a.model_a, ensemble_b.model_a, ignore_keys=["linear1.weight"])
        self._state_dicts_equal(model_b, ensemble_a.model_b, ensemble_b.model_b, ignore_keys=["linear1.weight"])
        interpolated = interpolate_dict(ensemble_a.state_dict(), ensemble_b.state_dict(), 0.5)
        params_interpolated = interpolate_dict(params_a, params_b, 0.5)
        self._state_dicts_equal(interpolated, ensemble_a, ensemble_b, ignore_keys=["model_a.linear1.weight", "model_b.linear1.weight"])
        self.assertTrue(torch.all(interpolated["model_a.linear1.weight"] == params_interpolated["linear1.weight"]))
        self.assertTrue(torch.all(interpolated["model_b.linear1.weight"] == params_interpolated["linear1.weight"]))

    def _make_submodel(self, n_in, n_out, h1, h2, sparsity=1., scale_narrow_to_wide_weights=0.):
        wide = self.TestModel(n_in, h1, h2, n_out)
        s1 = int(h1 * sparsity)
        s2 = int(h2 * sparsity)
        narrow = self.TestModel(n_in, s1, s2, n_out)
        narrow.linear1.weight.data = wide.linear1.weight[:s1, :]
        narrow.linear1.bias.data = wide.linear1.bias[:s1]
        narrow.linear2.weight.data = wide.linear2.weight[:s2, :s1]
        narrow.linear2.bias.data = wide.linear2.bias[:s2]
        narrow.linear3.weight.data = wide.linear3.weight[:, :s2]
        return wide, narrow

    def test_embed_into_wider_network_symmetry(self):
        narrow_wide_data = [
            (
                self.TestModel(20, 15, 10, 2),
                self.TestModel(20, 30, 20, 2),
                self._make_dataloader(torch.randn([100, 20])),
            ),
            (
                cifar_vgg.Model.get_model_from_name("cifar_vgg_11_4", initializer=kaiming_normal),
                cifar_vgg.Model.get_model_from_name("cifar_vgg_11_8", initializer=kaiming_normal),
                self._make_dataloader(torch.randn([16, 3, 32, 32])),
            ),
            (
                cifar_resnet.Model.get_model_from_name("cifar_resnet_20_4", initializer=kaiming_normal),
                cifar_resnet.Model.get_model_from_name("cifar_resnet_20_8", initializer=kaiming_normal),
                self._make_dataloader(torch.randn([16, 3, 32, 32])),
            ),
            (
                cifar_vgg.Model.get_model_from_name("cifar_vgg_11_4", initializer=kaiming_normal, batchnorm_type="layernorm"),
                cifar_vgg.Model.get_model_from_name("cifar_vgg_11_8", initializer=kaiming_normal, batchnorm_type="layernorm$2"),
                self._make_dataloader(torch.randn([16, 3, 32, 32])),
            ),
            (
                cifar_resnet.Model.get_model_from_name("cifar_resnet_20_4", initializer=kaiming_normal, batchnorm_type="layernorm"),
                cifar_resnet.Model.get_model_from_name("cifar_resnet_20_8", initializer=kaiming_normal, batchnorm_type="layernorm$2"),
                self._make_dataloader(torch.randn([16, 3, 32, 32])),
            ),
        ]
        for narrow, wide, dataloader in narrow_wide_data:
            with self.subTest(model=narrow):
                narrow_dict = narrow.state_dict()
                perm_spec = self._get_perm_spec(narrow_dict)
                embed_narrow_dict = perm_spec.apply_padding(
                    narrow_dict, perm_spec.get_sizes(wide.state_dict()))
                np.testing.assert_array_almost_equal(
                    evaluate_model(narrow, dataloader, device="cpu"),
                    evaluate_model(wide, dataloader, state_dict=to_torch_device(embed_narrow_dict, device="cpu"), device="cpu"))


    def test_align_embed_submodel(self):
        dataloader = self.models[0][1]
        for sparsity in [1, 0.5, 0.25, 0]:
            for align_class in [lambda x, y, z: PartialWeightAlignment(x, kernel="linear", order="random", target_sizes=y),
                            #   lambda x, y, z: PartialActivationAlignment(x, dataloader=dataloader, model=z, device="cpu", target_sizes=y),
                              ]:
                with self.subTest(sparsity=sparsity, align_class=align_class):
                    wide, narrow = self._make_submodel(20, 2, 20, 16, sparsity=sparsity)
                    # self._validate_symmetry(narrow, dataloader, lambda x: to_numpy(wide.state_dict()))
                    wide_dict = wide.state_dict()
                    perm_spec = self._get_perm_spec(wide_dict)
                    rand_perm = perm_spec.get_random_permutation(wide_dict)
                    wide_dict = perm_spec.apply_permutation(wide_dict, rand_perm)
                    sizes = perm_spec.get_sizes(wide_dict)
                    align_obj = align_class(perm_spec, sizes, wide)
                    perm, similarities = align_obj.fit(wide_dict, narrow.state_dict())
                    permuted, mask = align_obj.transform()
                    # check that dense permutation is recovered for nonzero sparse weights
                    for k, v in perm.items():
                        assert np.all((v == rand_perm[k])[v < sparsity * len(v)])
                    # check that sparse model is the same with and without padding
                    np.testing.assert_array_almost_equal(
                        evaluate_model(narrow, dataloader, device="cpu"),
                        evaluate_model(wide, dataloader, state_dict=to_torch_device(permuted, device="cpu"), device="cpu"))

    # def _make_partially_aligned(self, n_in, n_out, n_aligned, n_unaligned):
    #     full_h2 = n_aligned + 2 * n_unaligned
    #     full_h1 = full_h2 * 2
    #     full_model = self.TestModel(n_in, full_h1, full_h2, n_out)
    #     half_h2 = full_h2 - n_unaligned
    #     half_h1 = half_h2 * 2
    #     half_model = self.TestModel(n_in, half_h1, half_h2, n_out)
    #     ensemble = full_model.state_dict()
    #     params_a = {}
    #     params_b = {}
    #     for i, (k, v) in enumerate(ensemble.items()):
    #         if i in [0, 1]:  # crop outputs of layer 1 weights and biases
    #             params_a[k] = v[:half_h1]
    #             params_b[k] = v[-half_h1:]
    #         elif i == 2: # layer 2 weights, crop both inputs and outputs
    #             # A maps to A only, shared maps to everything, B maps to B only
    #             # therefore, set top right and bottom left corners to 0
    #             v[:n_unaligned, half_h1:] = 0
    #             print(v.shape, n_unaligned, half_h1)
    #             v[-n_unaligned:, :n_unaligned * 2] = 0
    #             ensemble[k] = v
    #             params_a[k] = v[:half_h2, :half_h1]
    #             params_b[k] = v[-half_h2:, -half_h1:]
    #         elif i in [3, 4]:  # crop outputs of layer 2 bias, layer 3 weights
    #             params_a[k] = v[..., :half_h2]
    #             params_b[k] = v[..., -half_h2:]
    #         else:  # layer 3 bias, copy directly
    #             params_a[k] = v
    #             params_b[k] = v
    #     full_model.load_state_dict(ensemble)
    #     out_weight_a = 1 - 0.5 * torch.tensor([False] * n_unaligned + [True] * n_aligned)
    #     out_weight_b = 1 - 0.5 * torch.tensor([True] * n_aligned + [False] * n_unaligned)
    #     return full_model, half_model, params_a, params_b, out_weight_a, out_weight_b

    # def test_ensemble_model(self):
    #     x = next(iter(self.sequential_models[0][1]))[0]
    #     with torch.no_grad():
    #         for n in [5, 0, 5, 10]:
    #             full_model, half_model, params_a, params_b, out_weight_a, out_weight_b = self._make_partially_aligned(20, 2, n, 10 - n)
    #             ensemble_model = EnsembleModel(half_model, params_a, params_b, out_weight_a, out_weight_b)
    #             y1 = full_model(x)
    #             y2 = ensemble_model(x) * 2
    #             print(n, y1, y2)
    #             # np.testing.assert_allclose(y1.detach().numpy(), y2.detach().numpy(), atol=1e-7)
    #             # ensemble_model = EnsembleModel.from_align_mask(half_model, params_a, params_b, out_weight_b)
    #             y = full_model.test_forward_1(x)
    #             y_a = ensemble_model.model_a.test_forward_1(x)
    #             y_b = ensemble_model.model_b.test_forward_1(x)
    #             z = full_model.test_forward_2(y)
    #             z_a = ensemble_model.model_a.test_forward_2(y_a)
    #             z_b = ensemble_model.model_b.test_forward_2(y_b)
    #             # print(full_model.state_dict(), params_a, params_b)
    #             # print("layer2",
    #             #     full_model.state_dict()["linear2.weight"][0],
    #             #     params_a["linear2.weight"][0],
    #             #     params_b["linear2.weight"][0],
    #             #     full_model.state_dict()["linear2.bias"],
    #             #     params_a["linear2.bias"],
    #             #     params_b["linear2.bias"]
    #             # )
    #             print("fc", z, z_a, z_b, z_a + z_b)  # torch.allclose(z_a + z_b, z)
    #             break

        # state_dict = model.state_dict()
        # perm_spec = self._get_perm_spec(state_dict)
        # align_obj = PartialWeightAlignment(perm_spec, target_sizes)
        # perms, mask = align_obj._split_perms(perms, sizes_a, sizes_b)

    def _get_scale_spec(self, state_dict):
        if any("block" in k for k in state_dict.keys()):
            return ScaleSpec.from_residual_model(state_dict)
        return ScaleSpec.from_sequential_model(state_dict)

    def test_scale(self):
        for model, data in self.models:
            with self.subTest(model=model):
                state_dict = model.state_dict()
                scale_spec = self._get_scale_spec(state_dict)

                identity_scale = scale_spec.get_identity_scale(state_dict)
                transform_fn = lambda x: scale_spec.apply_scale(x, identity_scale)
                self._validate_symmetry(model, data, transform_fn)

                const_scale = {k: np.full_like(v, 0.9) for k, v in identity_scale.items()}
                transform_fn = lambda x: scale_spec.apply_scale(x, const_scale)
                self._validate_symmetry(model, data, transform_fn)

                rand_scale = scale_spec.get_random_scale(state_dict)
                transform_fn = lambda x: scale_spec.apply_scale(x, rand_scale)
                self._validate_symmetry(model, data, transform_fn)

    def test_rollback_normalization(self):
        for model, data in self.models:
            if any("bn" in x for x in model.state_dict().keys()):
                with self.subTest(model=model):
                    state_dict = model.state_dict()
                    scale_spec = self._get_scale_spec(state_dict)
                    transform_fn = lambda x: scale_spec.apply_rollback_batchnorm(x)
                    self._validate_symmetry(model, data, transform_fn)

    def test_scale_norm(self):
        for model, data in self.models:
            if any("bn" in x for x in model.state_dict().keys()):
                with self.subTest(model=model):
                    state_dict = to_numpy(model.state_dict())
                    scale_spec = self._get_scale_spec(state_dict)

                    ones_state_dict = {k: np.ones_like(v) for k, v in state_dict.items()}
                    ones_norm = scale_spec.get_norm(ones_state_dict, normalize=True)
                    for k, v in ones_norm.items():
                        np.testing.assert_allclose(v, 1.), (k, v)

                    rand_scale = scale_spec.get_random_scale(state_dict)
                    scaled_dict = scale_spec.apply_scale(state_dict, rand_scale)
                    norm = scale_spec.get_norm(state_dict)
                    new_norm = scale_spec.get_norm(scaled_dict)
                    for k, v in norm.compose(new_norm.inverse()).items():
                        np.testing.assert_allclose(rand_scale[k], v)
                    avg_norm = scale_spec.get_avg_norm(state_dict, scaled_dict)

                    # new_state_dict = scale_spec.apply_scale(scale_spec.apply_scale(
                    #     state_dict, rand_scale), rand_scale.inverse())
                    # identity = rand_scale.compose(rand_scale.inverse()).to_matrices()
                    # for k in state_dict.keys():
                    #     np.testing.assert_allclose(state_dict[k], new_state_dict[k])
                    # for x in identity.values():
                    #     np.testing.assert_allclose(x, np.eye(len(x)))
                    # scale = scale_spec.get_norm(state_dict)
                    # scale_avg = scale_spec.get_avg_norm(state_dict, output)


if __name__ == "__main__":
    unittest.main()
