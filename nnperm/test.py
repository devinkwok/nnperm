import unittest
from copy import deepcopy
from itertools import product
import torch
import numpy as np
import torch
from torch import nn

import sys
from nnperm.utils import multiplicative_weight_noise
sys.path.append("open_lth")
from open_lth.models import cifar_resnet
from open_lth.models import cifar_vgg
from open_lth.models.initializers import kaiming_normal

from nnperm.perm import PermutationSpec
from nnperm.align import *
from nnperm.error import evaluate

from rebasin.torch_utils import torch_weight_matching


class TestNNPerm(unittest.TestCase):

    def make_dataloader(self, tensor):
        dataset = torch.utils.data.TensorDataset(tensor, torch.zeros(tensor.shape[0], dtype=torch.long))
        return torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    def setUp(self) -> None:
        ## Setup
        self.sequential_models = [
            (
                nn.Sequential(
                    nn.Linear(20, 10),
                    nn.ReLU(),
                    nn.Linear(10, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2),
                ),
                self.make_dataloader(torch.randn([10, 20])),
            ),
            (
                nn.Sequential(
                    nn.Conv2d(3, 10, 3),
                    nn.BatchNorm2d(10),
                    nn.ReLU(),
                    nn.Conv2d(10, 5, 3),
                    nn.BatchNorm2d(5),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool2d([1, 1]),
                    nn.Flatten(start_dim=1),
                    nn.Linear(5, 2),
                ),
                self.make_dataloader(torch.randn([10, 3, 9, 9])),
            ),
            (
                nn.Sequential(
                    cifar_vgg.Model.ConvModule(3, 64, batchnorm_type=None),
                    cifar_vgg.Model.ConvModule(64, 128, batchnorm_type=None),
                ),
                self.make_dataloader(torch.randn([10, 3, 32, 32])),
            ),
            (
                cifar_vgg.Model.get_model_from_name("cifar_vgg_11", initializer=kaiming_normal),
                self.make_dataloader(torch.randn([10, 3, 32, 32])),
            ),
        ]
        self.conv_models = [
            (
                nn.Sequential(
                    nn.Conv2d(3, 10, 3),
                    cifar_resnet.Model.Block(10, 5, downsample=True)
                ),
                self.make_dataloader(torch.randn([10, 3, 32, 32])),
            ),
            # following models take longer to run
            # (
            #     add_skip_weights_to_open_lth_resnet(cifar_resnet.Model.get_model_from_name(
            #           "cifar_resnet_14_4", initializer=kaiming_normal)),
            #     self.make_dataloader(torch.randn([10, 3, 32, 32])),
            # ),
        ]
        for model, _ in self.sequential_models:
            self.randomize_batchnorm_weights(model)
            self.insert_small_weights(model, 0.05)

    def randomize_batchnorm_weights(self, model):
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

    def insert_small_weights(self, model, probability, scale_factor=1e-2):
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
                assert torch.equal(self.original[k], v)

    def validate_symmetry(self, model, data, transform_fn):
        output = evaluate(model, data, device="cpu")
        state_dict = model.state_dict()
        with self.StateUnchangedContextManager(state_dict):
            transformed = transform_fn(state_dict)
            transformed_output = evaluate(model, data, state_dict=transformed, device="cpu")
            np.testing.assert_allclose(output, transformed_output, rtol=1e-4, atol=1e-4)

    def validate_perm_align(self, model, data, perm_spec, perm, sklearn_like_obj, noise_std=0., allowed_errors=0, allowed_loss=1e-4):
        state_dict = model.state_dict()
        with self.StateUnchangedContextManager(state_dict):
            permuted_state_dict = perm_spec.apply_permutation(state_dict, perm)
            permuted_state_dict = multiplicative_weight_noise(permuted_state_dict, noise_std)
            found_perm, losses = sklearn_like_obj.fit(permuted_state_dict, state_dict)
            transform_fn = lambda x: sklearn_like_obj.fit_transform(permuted_state_dict, x)
            self.validate_symmetry(model, data, transform_fn)
        self.assertTrue(keys_match(perm, found_perm))
        for k, v in perm.items():
            self.assertLessEqual(np.count_nonzero(v != found_perm[k]), allowed_errors)
            # self.assertLessEqual(np.max(losses[k]), allowed_loss)

    def test_apply(self):
        for model, _ in self.sequential_models:
            state_dict = model.state_dict()
            perm_spec = PermutationSpec.from_sequential_model(model.state_dict())
            sizes = perm_spec.get_sizes(state_dict)
            self.assertEqual(len(sizes), len(perm_spec.perm_to_axes))
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
        for model, data in self.sequential_models:
            state_dict = model.state_dict()
            perm_spec = PermutationSpec.from_sequential_model(state_dict)
            transform_fn = lambda x: perm_spec.apply_rand_perm(x)
            self.validate_symmetry(model, data, transform_fn)
            rand_perm = perm_spec.get_random_permutation(state_dict)
            new_state_dict = perm_spec.apply_permutation(perm_spec.apply_permutation(
                state_dict, rand_perm), rand_perm.inverse())
            identity = rand_perm.compose(rand_perm.inverse()).to_matrices()
            for k in state_dict.keys():
                np.testing.assert_array_equal(state_dict[k], new_state_dict[k])
            for x in identity.values():
                np.testing.assert_array_equal(x, np.eye(len(x)))

    def test_align(self):
        for seed in range(100, 300, 100):
            self.setUp()
            for model, data in self.sequential_models:
                state_dict = model.state_dict()
                perm_spec = PermutationSpec.from_sequential_model(state_dict)
                for weight_noise, loss, order, make_perm_fn in product(
                    [0., 1e-2, 1e-1], ["linear", "sqexp"], ["forward", "backward", "random"],
                    [perm_spec.get_identity_permutation, perm_spec.get_random_permutation],
                ):
                    print(model, weight_noise, loss, order, make_perm_fn)
                    align_obj = WeightAlignment(perm_spec, loss, seed=seed, order=order)
                    perm = make_perm_fn(state_dict)
                    self.validate_perm_align(model, data, perm_spec, perm, align_obj, noise_std=weight_noise)

    # def test_partial_align(self):
    #     #TODO need to pad model to validate_symmetry
    #     for model, data in self.sequential_models:
    #         state_dict = model.state_dict()
    #         perm_spec = PermutationSpec.from_sequential_model(state_dict)
    #         for loss, order, make_perm_fn in zip(
    #             ["mse", "dot"], ["forward", "random"],
    #             [perm_spec.get_identity_permutation, perm_spec.get_random_permutation],
    #         ):
    #             sizes = perm_spec.get_sizes(state_dict)
    #             min_size = min(list(sizes.values()))
    #             for i in range(min_size + 1):
    #                 pad_size = {k: v + i for k, v in sizes.items()}
    #                 align_obj = WeightAlignment(perm_spec, loss, order=order, target_sizes=pad_size)
    #                 perm = make_perm_fn(state_dict)
    #                 self.validate_perm_align(model, data, perm_spec, perm, align_obj,
    #                                         allowed_errors=i, allowed_loss=1.)

    # def test_bootstrap(self):
    #     for model, data in self.sequential_models:
    #         state_dict = model.state_dict()
    #         perm_spec = PermutationSpec.from_sequential_model(state_dict)
    #         for loss, order, make_perm_fn in product(
    #             ["mse", "dot"], ["forward", "backward", "random"],
    #             [perm_spec.get_identity_permutation, perm_spec.get_random_permutation],
    #         ):
    #             align_obj = WeightAlignment(perm_spec, loss, bootstrap=1000, order=order)
    #             perm = make_perm_fn(state_dict)
    #             self.validate_perm_align(model, data, perm_spec, perm, align_obj,
    #                                     allowed_error_rate=0.1, allowed_loss=1e18)


if __name__ == "__main__":
    unittest.main()
