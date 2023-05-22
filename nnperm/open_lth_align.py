# Compute the permutation between 2 open_lth checkpoints
# save permutations in a format that makes sense to open_lth
from pathlib import Path

from nnperm.align import WeightAlignment, ActivationAlignment, PartialActivationAlignment, PartialWeightAlignment
from nnperm.spec import PermutationSpec
from nnperm.utils import get_open_lth_ckpt, get_dataloader, get_device, prune


# TODO temporary hack for layernorm
def _get_target_size_model(model, target_size_ckpt):
    source_size = model.state_dict()
    if target_size_ckpt is not None:
        _, _, size_params = get_open_lth_ckpt(target_size_ckpt)
        # make sure sizes differ by constant ratio
        ratio = None
        for k, v in size_params.items():
            if "layernorm" in k:
                new_ratio = v.shape[0] / source_size[k].shape[0]
                assert ratio is None or ratio == new_ratio
                ratio = new_ratio
        # scale mean/std of layernorm appropriately so they have the correct scale from the source network
        print(f"Scaling layernorm by {ratio} due to added padding")
        _, model, _ = get_open_lth_ckpt(target_size_ckpt, layernorm_scaling=ratio)
    return model


## Setup
def open_lth_align(ckpt_a: Path,
        ckpt_b: Path,
        type: str = "weight_linear",
        seed: int = 42,
        batch_size: int = 400,
        exclude: str = None,
        overwrite: bool = False,
        save_inverse: bool = False,
        target_size_ckpt_a: Path = None,
        target_size_ckpt_b: Path = None,
        prune_type: str = 'sparse_global',
        prune_randomize: str = 'identity',
        prune_fraction: float = 0.,
        verbose: bool = False,
):

    def name_from_path(open_lth_path):
        epoch = open_lth_path.stem.split("model_")[1]
        branch = open_lth_path.parent
        level = branch.parent
        replicate = level.parent
        experiment = replicate.parent
        path_str = "-".join([experiment.stem, replicate.stem, level.stem, branch.stem, epoch])
        if prune_fraction > 0:
            path_str += "-" + "-".join([prune_type, prune_randomize, str(prune_fraction)])
        return f"perm_{type}-{path_str}.pt"

    b2a_save_file = ckpt_b.parent / name_from_path(ckpt_a)
    a2b_save_file = ckpt_a.parent / name_from_path(ckpt_b)
    # skip if files already exist
    if b2a_save_file.exists() and not overwrite:
        if not save_inverse:
            print(f"File already exists {b2a_save_file}")
            return None, b2a_save_file
        elif a2b_save_file.exists():
            print(f"Files already exist {b2a_save_file}, {a2b_save_file}")
            return a2b_save_file, b2a_save_file

    exclude_layers = exclude.split(",") if exclude is not None else None
    # get model and data
    (model_hparams_a, dataset_hparams_a), model_a, params_a = get_open_lth_ckpt(ckpt_a)
    (model_hparams_b, dataset_hparams_b), model_b, params_b = get_open_lth_ckpt(ckpt_b)
    # prune model before alignment
    if prune_fraction > 0.:
        model_a, one_shot_mask_a = prune(model_a, prune_fraction, type=prune_type, randomize=prune_randomize, seed=seed)
        params_a = model_a.state_dict()
        model_b, one_shot_mask_b = prune(model_b, prune_fraction, type=prune_type, randomize=prune_randomize, seed=seed)
        params_b = model_b.state_dict()

    # get perm spec
    if "resnet" in model_hparams_a.model_name:
        print("Aligning residual model")
        perm_spec = PermutationSpec.from_residual_model(params_a)
    else:
        print("Aligning sequential model")
        perm_spec = PermutationSpec.from_sequential_model(params_a)
    if verbose:
        print(model_hparams_a.display)
        print(dataset_hparams_a.display)

    align_type, kernel, *other_args = type.split("_")
    perm_spec = perm_spec.subset(exclude_axes=exclude_layers)

    AlignClass = ActivationAlignment if align_type == "activation" else WeightAlignment
    if any(x != y for x, y in zip(perm_spec.get_sizes(params_a).values(), perm_spec.get_sizes(params_b).values())):
        AlignClass = PartialActivationAlignment if align_type == "activation" else PartialWeightAlignment
        # get the correct size of model to embed into
        model_a = _get_target_size_model(model_a, target_size_ckpt_a)
        model_b = _get_target_size_model(model_b, target_size_ckpt_b)
        print(f"Sizes differ, using {AlignClass.__name__}")  # if sizes differ, use partial alignment
    if align_type == "activation":
        dataset_name = other_args[0]
        n_train = int(other_args[1])
        intermediate_type = "last" if len(other_args) < 3 else other_args[2]
        if dataset_name == dataset_hparams_a.dataset_name:
            dataset_hparams = dataset_hparams_a
        elif dataset_name == dataset_hparams_b.dataset_name:
            dataset_hparams = dataset_hparams_b
        else:
            raise ValueError(f"Dataset must be either from A: {dataset_hparams_a.dataset_name}, or B: {dataset_hparams_b.dataset_name}")
        dataloader = get_dataloader(dataset_hparams, n_train, train=True, batch_size=batch_size)
        align_obj = AlignClass(
            perm_spec,
            model_a=model_a,
            model_b=model_b,
            dataloader=dataloader,
            intermediate_type=intermediate_type,
            exclude=exclude_layers,
            kernel=kernel,
            verbose=False,
            device=get_device(),
        )
    else:
        align_obj = AlignClass(
                    perm_spec,
                    kernel=kernel,
                    init_perm=None,
                    max_iter=100,
                    seed=seed,
                    order="random",
                    verbose=False)

    perm, align_loss = align_obj.fit(params_a, params_b)

    # save a copy of each permutation to ckpt_a and ckpt_b locations
    perm_spec.save_to_file(perm, b2a_save_file)
    if save_inverse:
        perm_spec.save_to_file(perm.inverse(), a2b_save_file)
        print(f"Saved to {b2a_save_file} and {a2b_save_file}.")
        return a2b_save_file, b2a_save_file
    else:
        print(f"Saved to {b2a_save_file}.")
        return None, b2a_save_file
