from copy import deepcopy
import unittest
import torch
import numpy as np
import torch
from torch import conv1d, nn

import sys
sys.path.append("open_lth")
from open_lth.models import cifar_resnet
from open_lth.models import cifar_vgg
from open_lth.models.initializers import kaiming_normal

from nnperm_utils import evaluate_per_sample, multiplicative_weight_noise, add_skip_weights_to_open_lth_resnet
import nnperm_old as old
import nnperm as new

from rebasin.torch_utils import torch_weight_matching


class TestPermuteNN(unittest.TestCase):

    def make_dataloader(self, tensor):
        dataset = torch.utils.data.TensorDataset(tensor, torch.zeros(tensor.shape[0]))
        return torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    def setUp(self) -> None:
        ## Setup
        self.mlp_models = [
            (
                nn.Sequential(
                    nn.Linear(20, 10),
                    nn.ReLU(),
                    nn.Linear(10, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2),
                ),
                self.make_dataloader(torch.randn([10, 20])),
                [
                    np.full(10, 0.5),
                    np.random.randn(5)**2,
                    None,
                ],
                [
                    np.random.permutation(10),
                    np.random.permutation(5),
                    None,
                ],
            ),
        ]
        self.conv_models = [
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
            # (
            #     nn.Sequential(
            #         nn.Conv2d(3, 10, 3),
            #         cifar_resnet.Model.Block(10, 5, downsample=True)
            #     ),
            #     self.make_dataloader(torch.randn([10, 3, 32, 32])),
            # ),
            (
                nn.Sequential(
                    cifar_vgg.Model.ConvModule(3, 64),
                    cifar_vgg.Model.ConvModule(64, 128),
                ),
                self.make_dataloader(torch.randn([10, 3, 32, 32])),
            ),
            # following models take longer to run
            # (
            #     add_skip_weights_to_open_lth_resnet(cifar_resnet.Model.get_model_from_name(
            #           "cifar_resnet_14_4", initializer=kaiming_normal)),
            #     self.make_dataloader(torch.randn([10, 3, 32, 32])),
            # ),
            (
                cifar_vgg.Model.get_model_from_name("cifar_vgg_11", initializer=kaiming_normal),
                self.make_dataloader(torch.randn([10, 3, 32, 32])),
            ),
        ]
        for model, data in self.conv_models:
            self.randomize_batchnorm_weights(model)
            self.insert_small_weights(model, 0.05)

    def randomize_batchnorm_weights(self, model):
        state_dict = model.state_dict()
        with torch.no_grad():
            for k, v in state_dict.items():
                if new._is_batchnorm(k, v):
                    key = k[:-len(".weight")]
                    tensors = [(k, v),
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
                torch.testing.assert_equal(self.original[k], v)

    def validate_symmetry(self, transform_fn, model, data):
        output = evaluate_per_sample(model, data, device="cpu")
        state_dict = deepcopy(model.state_dict())
        with self.StateUnchangedContextManager(state_dict):
            normalized = transform_fn(state_dict)
            normalized_output = evaluate_per_sample(model, data, state_dict=normalized, device="cpu")
            diff = np.abs(output - normalized_output)
            print("Testing", transform_fn.__name__, ": max abs diff", np.max(diff), "mean abs diff", np.mean(diff))
            np.testing.assert_allclose(output, normalized_output, rtol=1e-4, atol=1e-4)

    def validate_scaling(self, model, scale):
        state_dict = deepcopy(model.state_dict())
        normalized = new.canonical_normalization(state_dict)[0]
        with self.StateUnchangedContextManager(normalized):
            scaled_dict = new.scale_state_dict(normalized, scale)
            for s_1, s_2 in zip(new.inverse_scale(scale), new.get_normalizing_scale(scaled_dict)):
                if s_1 is None or s_2 is None:
                    self.assertIsNone(s_1)
                    self.assertIsNone(s_2)
                else:
                    np.testing.assert_allclose(np.array(s_1), np.array(s_2), rtol=1e-4, atol=1e-4)
            print("Testing scales, scales match")

    def test_mlp_normalization(self):
        # test scaling for mlp
        for model, data, scale, _ in self.mlp_models:
            self.validate_symmetry(old.canonical_renormalization, model, data)
            self.validate_symmetry(lambda x: new.scale_state_dict(x, scale), model, data)
            self.validate_symmetry(new.normalize_batchnorm, model, data)
            self.validate_symmetry(lambda x: new.canonical_normalization(x)[0], model, data)
            self.validate_scaling(model, scale)
            scales = new.get_normalizing_scale(model.state_dict())
            print('\nScale magnitude\n', [(torch.min(s).item(), torch.mean(s).item(), torch.max(s).item()) if s is not None else None for s in scales])

    def test_conv_normalization(self):
        # test scaling for convnet
        for model, data in self.conv_models:
            scale = new.random_scale(model.state_dict())
            self.validate_symmetry(lambda x: new.scale_state_dict(x, scale), model, data)
            self.validate_symmetry(new.normalize_batchnorm, model, data)
            self.validate_symmetry(lambda x: new.canonical_normalization(x)[0], model, data)
            # self.validate_symmetry(lambda x: new.canonical_normalization(x, align_shortcut="shortcut")[0], model, data)
            # self.validate_symmetry(lambda x: new.canonical_normalization(x, align_shortcut="block")[0], model, data)
            self.validate_scaling(model, scale)

    def validate_permutation_finder(self, finder_fn, model, permutations):
        state_dict = deepcopy(model.state_dict())
        # with self.StateUnchangedContextManager(state_dict):
        layer_names = filter(lambda x: "weight" in x, state_dict.keys())
        permuted_state_dict = new.permute_state_dict(state_dict, permutations)
        s_1, s_2, diffs = finder_fn(permuted_state_dict, state_dict)
        found_permutations = new.compose_permutation(s_2, new.inverse_permutation(s_1))
        self.assertEqual(len(permutations), len(found_permutations))
        for x, y, d, k in zip(permutations, found_permutations, diffs, layer_names):
            if x is None:
                self.assertIsNone(y)
            else:
                print("Found permutation", x.shape, "for layer", k)
                self.assertLess(torch.min(d), 1e-1)
        # check that permutations are valid
        for a, b in zip(s_1, s_2):
            if a is not None:
                np.testing.assert_array_equal(np.arange(len(a)), np.sort(a))
                np.testing.assert_array_equal(np.arange(len(b)), np.sort(b))

    def test_mlp_permutation(self):
        for model, data, _, perm in self.mlp_models:
            self.validate_symmetry(lambda x: old.permutate_state_dict_mlp(x, perm), model, data)
            self.validate_symmetry(lambda x: new.permute_state_dict(x, perm), model, data)
            self.validate_permutation_finder(old.find_permutations, model, perm)
            self.validate_permutation_finder(lambda x, y: new.get_normalizing_permutation(x, y), model, perm)

    def test_conv_permutation(self):
        for model, data in self.conv_models:
            perm = new.random_permutation(model.state_dict())
            self.validate_symmetry(lambda x: new.permute_state_dict(x, perm), model, data)
            self.validate_permutation_finder(lambda x, y: new.get_normalizing_permutation(x, y,), model, perm)
            # self.validate_permutation_finder(lambda x, y: new.get_normalizing_permutation(x, y,, align_shortcut="shortcut"), model, perm)
            # self.validate_permutation_finder(lambda x, y: new.get_normalizing_permutation(x, y,, align_shortcut="block"), model, perm)

    def test_random_transform(self):
        for model, data, scale, perm in self.mlp_models:
            scale = new.random_scale(model.state_dict(), n_layers=1)
            self.assertEqual(len(scale), 3)
            self.assertEqual(scale[0].shape, (10,))
            self.assertIsNone(scale[1])
            self.assertIsNone(scale[2])
            permutation = new.random_permutation(model.state_dict(), n_layers=2)
            self.assertEqual(len(permutation), 3)
            self.assertIsNotNone(permutation[0])
            self.assertIsNotNone(permutation[1])
            self.assertIsNone(permutation[2])

    def test_weight_noise(self):

        def assert_param_noise(state_dict, is_noisy):
            has_noise = [torch.any(v != 1).item() for v in state_dict.values()]
            np.testing.assert_array_equal(has_noise, is_noisy)

        for model, _, _, _ in self.mlp_models:
            state_dict = model.state_dict()
            n_params = len(state_dict)
            for k, v in state_dict.items():
                state_dict[k] = torch.ones_like(v)
            noisy = multiplicative_weight_noise(state_dict, 0.5)
            assert_param_noise(noisy, [True] * n_params)
            noisy = multiplicative_weight_noise(state_dict, 0.5, n_layers=1)
            assert_param_noise(noisy, [True] + [False] * (n_params - 1))
            noisy = multiplicative_weight_noise(state_dict, 0.5, include_keywords=["weight"])
            assert_param_noise(noisy, [True if "weight" in k else False for k in state_dict.keys()])
            noisy = multiplicative_weight_noise(state_dict, 0.5, exclude_keywords=["bias"])
            assert_param_noise(noisy, [False if "bias" in k else True for k in state_dict.keys()])
            noisy = multiplicative_weight_noise(state_dict, 0.5,
                include_keywords=["weight"], exclude_keywords=["0"])
            assert_param_noise(noisy, [True if "weight" in k and "0" not in k else False for k in state_dict.keys()])
            noisy = multiplicative_weight_noise(state_dict, 0.5, n_layers=2, include_keywords=["weight"])
            assert_param_noise(noisy, [True, False, True] + [False] * (n_params - 3))

    def test_resnet(self):
        model = cifar_resnet.Model.get_model_from_name("cifar_resnet_8_4", initializer=kaiming_normal)
        data = self.make_dataloader(torch.randn([10, 3, 32, 32]))
        output = evaluate_per_sample(model, data, device="cpu")
        with self.StateUnchangedContextManager(model.state_dict()):
            model_skip = add_skip_weights_to_open_lth_resnet(model)
            normalized_output = evaluate_per_sample(model_skip, data, device="cpu")
            diff = np.abs(output - normalized_output)
            print("Testing add_skip_weights_to_open_lth_resnet: max abs diff", np.max(diff), "mean abs diff", np.mean(diff))
            np.testing.assert_allclose(output, normalized_output, atol=1e-5)

    def test_sinkhorn(self):
        for i, (model, _) in enumerate(self.conv_models):
            model_1 = deepcopy(model.state_dict())
            self.setUp()
            model_2 = self.conv_models[i][0].state_dict()
            new.get_normalizing_permutation(model_1, model_2)

    def test_are_resnet_blocks_interchangeable_if_zero(self):
        model = cifar_resnet.Model.get_model_from_name(
                      "cifar_resnet_14_4", initializer=kaiming_normal)
        data = self.make_dataloader(torch.randn([10, 3, 32, 32]))
        # zero out block 0 weights
        state_dict = model.state_dict()
        state_dict["blocks.0.conv1.weight"].data.zero_()
        model.load_state_dict(state_dict)
        # swap blocks 0 and 1
        def swap_blocks(state_dict, i=0, j=1):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith(f"blocks.{i}"):
                    new_key = f"blocks.{j}" + k[len(f"blocks.{i}"):]
                elif k.startswith(f"blocks.{j}"):
                    new_key = f"blocks.{i}" + k[len(f"blocks.{j}"):]
                new_state_dict[new_key] = v
            return new_state_dict
        self.validate_symmetry(swap_blocks, model, data)

    def test_git_rebasin_weight_matching(self):
        for model, data, scale, perm in self.mlp_models:
            self.validate_permutation_finder(torch_weight_matching, model, perm)
        for model, data in self.conv_models:
            perm = new.random_permutation(model.state_dict())
            self.validate_permutation_finder(torch_weight_matching, model, perm)



if __name__ == "__main__":
    unittest.main()
