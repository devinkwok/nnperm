import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torchvision

import sys
sys.path.append("open_lth")
from open_lth.models import registry
from open_lth.foundations.hparams import ModelHparams

from nnperm_utils import calculate_errors, evaluate_per_sample
from nnperm import canonical_normalization, canonical_permutation, geometric_realignment, permute_state_dict, random_transform
 
## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, type=Path)
parser.add_argument('--n_replicates', required=True, type=int)
parser.add_argument('--barrier_resolution', default=10, type=int)
parser.add_argument('--test_points', default=500, type=int)
parser.add_argument('--ckpt_root', default="../../open_lth_data/", type=Path)
args = parser.parse_args()

# get open_lth hparams
with open(args.ckpt_root / args.ckpt / f"replicate_1/main/hparams.log", 'r') as f:
    hparam_lines = f.readlines()
hparams = {}
for line in hparam_lines:
    line = line.strip()
    if line.endswith(" Hyperparameters"):
        header = line[:-len(" Hyperparameters")]
        hparams[header] = {}
    elif line.startswith("* "):
        k, v = line[len("* "):].split(" => ")
        hparams[header][k] = v
    else:
        raise ValueError(line)
print(hparams)

# get model and data
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
])
dataset = torchvision.datasets.MNIST(root="../../open_lth_datasets/mnist/", train=True, download=False, transform=transforms)
dataset = torch.utils.data.Subset(dataset, np.arange(args.test_points))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.test_points, shuffle=False)
model = registry.get(ModelHparams(
    hparams["Model"]["model_name"],
    hparams["Model"]["model_init"],
    hparams["Model"]["batchnorm_init"])).to(device="cuda")

### Experiment: change in error barrier on 2 randomly trained networks after realigning weights

def load_ckpt(replicate):
    path = Path(args.ckpt_root / args.ckpt)
    ckpt = torch.load(path / f"replicate_{replicate}/main/checkpoint.pth")
    return deepcopy(ckpt["model_state_dict"])

def get_errors(state_dict_f, state_dict_g):
    return calculate_errors(model, state_dict_f, state_dict_g,
                    dataloader, n_samples=args.barrier_resolution)

save_dir = Path(f"outputs/{args.ckpt}_{args.n_replicates}_{args.barrier_resolution}_{args.test_points}")
save_dir.mkdir(parents=True, exist_ok=True)
for i in range(1, args.n_replicates + 1):
    for j in range(1, i):
        print(f"Computing error barriers for {i}, {j}")
        state_dict_f = load_ckpt(i)
        state_dict_g = load_ckpt(j)
        normalized_f, scale_f = canonical_normalization(state_dict_f)
        normalized_g, scale_g = canonical_normalization(state_dict_g)
        s_f, s_g, loss = geometric_realignment(state_dict_f, state_dict_g)
        permuted_f = permute_state_dict(state_dict_f, s_f)
        permuted_g = permute_state_dict(state_dict_g, s_g)
        np_s_f, np_s_g, np_loss = geometric_realignment(normalized_f, normalized_g)
        norm_and_perm_f = permute_state_dict(normalized_f, np_s_f)
        norm_and_perm_g = permute_state_dict(normalized_g, np_s_g)
        torch.save({
                "scale_f": s_f,
                "scale_g": s_g,
                "perm_f": s_f,
                "perm_g": s_g,
                "perm_loss": loss,
                "norm_and_perm_f": np_s_f,
                "norm_and_perm_g": np_s_g,
                "norm_and_perm_loss": np_loss,
                "original_barriers": get_errors(state_dict_f, state_dict_g),
                "normalized_barriers": get_errors(normalized_f, normalized_g),
                "permuted_barriers": get_errors(permuted_f, permuted_g),
                "norm_and_perm_barriers": get_errors(norm_and_perm_f, norm_and_perm_g),
            }, save_dir / f"errors_{i}_{j}.pt")

        ## sanity check
        # random_f = random_transform(state_dict_f, scale=False)
        # values_f = evaluate_per_sample(model, dataloader, state_dict=state_dict_f)
        # values_g = evaluate_per_sample(model, dataloader, state_dict=state_dict_g)
        # norm_f = evaluate_per_sample(model, dataloader, state_dict=normalized_f)
        # perm_f = evaluate_per_sample(model, dataloader, state_dict=permuted_f)
        # rand_f = evaluate_per_sample(model, dataloader, state_dict=random_f)
        # print(np.mean((values_f - values_g)**2))
        # print(np.mean((values_f - norm_f)**2))
        # print(np.mean((values_f - perm_f)**2))
        # print(np.mean((values_f - rand_f)**2))
        # torch.save(state_dict_f, "state_dict_f_weights.pt")
        # torch.save(permuted_f, "permuted_f_weights.pt")
        # torch.save(random_f, "random_f_weights.pt")
