from copy import deepcopy
import unittest
import torch
import numpy as np
import torch
from torch import nn

from nnperm_utils import evaluate_per_sample
import nnperm_old as old
import nnperm as new


class TestPermuteNN(unittest.TestCase):

    def make_dataloader(self, tensor):
        dataset = torch.utils.data.TensorDataset(tensor, torch.zeros(tensor.shape[0]))
        return torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    def setUp(self) -> None:
        import sys
        sys.path.append("open_lth")
        from open_lth.models.mnist_lenet import Model
        from open_lth.models.initializers import kaiming_normal
        ## Setup
        self.data = self.make_dataloader(torch.randn([10, 1, 28, 28]))
        self.model = Model.get_model_from_name("mnist_lenet_20_10", initializer=kaiming_normal)
        self.perm = [np.random.permutation(v.shape[0]) for k, v in self.model.state_dict().items() if "weight" in k]
        self.perm[-1] = None

        a = 10  # hidden outputs, layer 1
        b = 5  # hidden outputs, layer 2
        self.mlp_data = self.make_dataloader(torch.randn([10, 20]))
        self.mlp_model = nn.Sequential(
                nn.Linear(20, a),
                nn.ReLU(),
                nn.Linear(a, b),
                nn.ReLU(),
                nn.Linear(b, 2),
            )
        self.mlp_scale = [
            np.full(a, 0.5),
            np.random.randn(b)**2,
            1,
        ]
        self.mlp_perm = [
            np.random.permutation(a),
            np.random.permutation(b),
            None,
        ]
        self.conv_data = self.make_dataloader(torch.randn([10, 3, 9, 9]))
        self.conv_model = nn.Sequential(
                nn.Conv2d(3, a, 3),
                nn.ReLU(),
                nn.BatchNorm2d(a),
                nn.Conv2d(a, b, 3),
                nn.ReLU(),
                nn.BatchNorm2d(b),
                nn.AdaptiveMaxPool2d([1, 1]),
                nn.Flatten(start_dim=1),
                nn.Linear(b, 2),
            )
        self.conv_scale = [
            np.full(a, 0.5),
            np.random.randn(b)**2,
            1,
        ]
        self.conv_perm = [
            np.random.permutation(a),
            np.random.permutation(b),
            None,
        ]

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
            np.testing.assert_allclose(output, normalized_output, atol=1e-5)

    def validate_scaling(self, model, scale):
        state_dict = deepcopy(model.state_dict())
        normalized = new.canonical_normalization(state_dict)
        with self.StateUnchangedContextManager(normalized):
            scaled_dict = new.scale_state_dict(normalized, scale)
            for s_1, s_2 in zip(new.inverse_scale(scale), new.get_normalizing_scale(scaled_dict)):
                np.testing.assert_allclose(np.array(s_1), np.array(s_2), atol=1e-3)
            print("Testing scales, scales match")

    def test_mlp_normalization(self):
        # test scaling for mlp
        self.validate_symmetry(old.canonical_renormalization, self.mlp_model, self.mlp_data)
        self.validate_symmetry(lambda x: new.scale_state_dict(x, self.mlp_scale), self.mlp_model, self.mlp_data)
        self.validate_symmetry(new.normalize_batchnorm, self.mlp_model, self.mlp_data)
        self.validate_symmetry(new.canonical_normalization, self.mlp_model, self.mlp_data)
        self.validate_scaling(self.mlp_model, self.mlp_scale)

    def test_conv_normalization(self):
        # test scaling for convnet
        self.validate_symmetry(lambda x: new.scale_state_dict(x, self.conv_scale), self.conv_model, self.conv_data)
        self.validate_symmetry(new.normalize_batchnorm, self.conv_model, self.conv_data)
        self.validate_symmetry(new.canonical_normalization, self.conv_model, self.conv_data)
        self.validate_scaling(self.conv_model, self.conv_scale)

    def validate_permutation_finder(self, finder_fn, model, permutations):
        state_dict = deepcopy(model.state_dict())
        # with self.StateUnchangedContextManager(state_dict):
        layer_names = filter(lambda x: "weight" in x, state_dict.keys())
        permuted_state_dict = new.permute_state_dict(state_dict, permutations)
        found_permutations = finder_fn(permuted_state_dict, state_dict)
        s_1, s_2, diffs = list(zip(*found_permutations))
        found_permutations = new.compose_permutation(s_2, new.inverse_permutation(s_1))
        self.assertEqual(len(permutations), len(found_permutations))
        for x, y, d, k in zip(permutations, found_permutations, diffs, layer_names):
            if x is None:
                self.assertIsNone(y)
            else:
                if np.any(x != y.numpy()):
                    self.assertFalse("Failed to find permutation for layer", k, d.item(), x, y)
                else:
                    print("Found permutation", x, "for layer", k)
                    self.assertLess(abs(d), 1e-15)

    def test_mlp_permutation(self):
        self.validate_symmetry(lambda x: old.permutate_state_dict_mlp(x, self.mlp_perm), self.mlp_model, self.mlp_data)
        self.validate_symmetry(lambda x: new.permute_state_dict(x, self.mlp_perm), self.mlp_model, self.mlp_data)
        self.validate_permutation_finder(old.find_permutations, self.mlp_model, self.mlp_perm)
        self.validate_permutation_finder(new.geometric_realignment, self.mlp_model, self.mlp_perm)
        self.validate_permutation_finder(new.geometric_realignment, self.mlp_model, self.mlp_perm)

        ## these are more expensive to run:
        # self.validate_symmetry(lambda x: new.permute_state_dict(x, self.perm), self.model, self.data)
        # self.validate_permutation_finder(new.geometric_realignment, self.model, self.perm)

    def test_conv_permutation(self):
        self.validate_symmetry(lambda x: new.permute_state_dict(x, self.conv_perm), self.conv_model, self.conv_data)
        self.validate_permutation_finder(new.geometric_realignment, self.conv_model, self.conv_perm)


if __name__ == "__main__":
    unittest.main()
