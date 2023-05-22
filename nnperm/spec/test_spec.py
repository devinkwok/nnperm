import unittest
from collections import defaultdict
import numpy as np

import sys
sys.path.append("open_lth")
from open_lth.models import cifar_resnet
from open_lth.models.initializers import kaiming_normal

from nnperm.spec import PermutationSpec


class TestModelSpec(unittest.TestCase):

    """permutation_spec_from_axes_to_perm and resnet20_permutation_spec adapted from
    https://github.com/samuela/git-re-basin
    Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2022). Git re-basin: Merging models modulo permutation symmetries. arXiv preprint arXiv:2209.04836.
    """
    @staticmethod
    def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
        perm_to_axes = defaultdict(list)
        for wk, axis_perms in axes_to_perm.items():
            axes = []
            for axis, perm in enumerate(axis_perms):
                if perm is not None:
                    is_input = (axis != len(axis_perms) - 1)
                    perm_to_axes[perm].append((wk, axis, is_input))
                    axes.append((perm, is_input))
                else:
                    axes.append(perm)
                axes_to_perm[wk] = tuple(axes)
        return PermutationSpec(group_to_axes=dict(perm_to_axes), axes_to_group=axes_to_perm)

    @staticmethod
    def resnet20_permutation_spec() -> PermutationSpec:
        conv = lambda name, p_in, p_out: {f"{name}/kernel": (None, None, p_in, p_out)}
        norm = lambda name, p: {f"{name}/scale": (p, ), f"{name}/bias": (p, )}
        dense = lambda name, p_in, p_out: {f"{name}/kernel": (p_in, p_out), f"{name}/bias": (p_out, )}

        # This is for easy blocks that use a residual connection, without any change in the number of channels.
        easyblock = lambda name, p: {
            **conv(f"{name}/conv1", p, f"P_{name}_inner"),
            **norm(f"{name}/norm1", f"P_{name}_inner"),
            **conv(f"{name}/conv2", f"P_{name}_inner", p),
            **norm(f"{name}/norm2", p)
        }

        # This is for blocks that use a residual connection, but change the number of channels via a Conv.
        shortcutblock = lambda name, p_in, p_out: {
            **conv(f"{name}/conv1", p_in, f"P_{name}_inner"),
            **norm(f"{name}/norm1", f"P_{name}_inner"),
            **conv(f"{name}/conv2", f"P_{name}_inner", p_out),
            **norm(f"{name}/norm2", p_out),
            **conv(f"{name}/shortcut/layers_0", p_in, p_out),
            **norm(f"{name}/shortcut/layers_1", p_out),
        }

        return TestModelSpec.permutation_spec_from_axes_to_perm({
            **conv("conv1", None, "P_bg0"),
            **norm("norm1", "P_bg0"),
            #
            **easyblock("blockgroups_0/blocks_0", "P_bg0"),
            **easyblock("blockgroups_0/blocks_1", "P_bg0"),
            **easyblock("blockgroups_0/blocks_2", "P_bg0"),
            #
            **shortcutblock("blockgroups_1/blocks_0", "P_bg0", "P_bg1"),
            **easyblock("blockgroups_1/blocks_1", "P_bg1"),
            **easyblock("blockgroups_1/blocks_2", "P_bg1"),
            #
            **shortcutblock("blockgroups_2/blocks_0", "P_bg1", "P_bg2"),
            **easyblock("blockgroups_2/blocks_1", "P_bg2"),
            **easyblock("blockgroups_2/blocks_2", "P_bg2"),
            #
            **dense("dense", "P_bg2", None),
        })

    class TranslatedKeyMatch(dict):
        def assert_equal(self, a, b, allow_new=True):
            if a not in self:
                if allow_new:
                    self[a] = b
                    return
                else:
                    assert False, f"missing key {a} when matching {b}"
            else:
                assert self[a] == b, f"mismatch {a} translated as {self[a]}, {b}"

    def assert_dict_lengths_equal(self, a, b):
        self.assertEqual(len(list(a.keys())), len(list(b.keys())))

    def test_resnet_spec(self):
        resnet = cifar_resnet.Model.get_model_from_name("cifar_resnet_20_64", initializer=kaiming_normal, batchnorm_type="layernorm")
        sd = resnet.state_dict()
        # use hard-coded git-re-basin resnet spec as reference
        ps = TestModelSpec.resnet20_permutation_spec()
        # resnet spec from nnperm
        ps2 = PermutationSpec.from_residual_model(sd)

        translate_layers = self.TranslatedKeyMatch()
        translate_perms = self.TranslatedKeyMatch()
        translate_axes = self.TranslatedKeyMatch()
        for (k, v), (k2, v2) in zip(ps.axes_to_group.items(), ps2.axes_to_group.items()):
            translate_layers.assert_equal(k, k2)
            for (i, x), (j, x2) in zip(reversed(list(enumerate(v))), enumerate(v2)):
                if x is None:
                    self.assertEqual(x, x2)
                else:
                    translate_perms.assert_equal(x[0], x2[0])
                    self.assertEqual(x[1], x2[1])
                    translate_axes.assert_equal((k, i), j)
        self.assert_dict_lengths_equal(ps.axes_to_group, ps2.axes_to_group)
        for (p, x), (p2, x2) in zip(ps.group_to_axes.items(), ps2.group_to_axes.items()):
            translate_perms.assert_equal(p, p2, False), (p, p2)
            for (k, d, i), (k2, d2, i2) in zip(x, x2):
                translate_layers.assert_equal(k, k2, False)  \
                    and translate_axes.assert_equal((k, d), d2, False) and i == i2, (k, d, i, k2, d2, i2)
            self.assertEqual(len(x), len(x2))
        self.assert_dict_lengths_equal(ps.group_to_axes, ps2.group_to_axes)


if __name__ == "__main__":
    unittest.main()
