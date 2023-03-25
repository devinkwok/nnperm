# Neural network permutation symmetries and linear mode connectivity


## Requirements

Uses `open_lth` to train and run models.
Clone repository and run `git submodule add https://github.com/devinkwok/open_lth`.
Install requirements using `pip install -r requirements.txt` (if using slurm scripts, this is done automatically).


## Usage

To train models for analysis, run `scripts/train-models.sh`.

To run experiments, run `scripts/align-all.sh`.

See `plots/` for results.

The `nnperm.align.WeightAlignment` object and its inherited classes have a sci-kit learn style interface.
Call `fit()` and `transform()` to find and apply weight alignments.
Computed properties such as `perms_` and `similarity_` are appended with a `_`.

The permutations appropriate for a given model architecture are recorded in a `PermutationSpec` object.
This object is based on code from Ainsworth, Hayase, & Srinivasa (2022) at [https://github.com/samuela/git-re-basin](https://github.com/samuela/git-re-basin).


## Development

Permutations and scaling in `nnperm.py`.
`PermutationSpec` and part of the `fit()` algorithm are based on Ainsworth, Hayase, & Srinivasa (2022).
Original implementation of geometric realignment (a different depreciated algorithm) by Udbhav Bamba.

To run unit tests, call `python -m nnperm.test`.


## Citations

Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2022). Git re-basin: Merging models modulo permutation symmetries. arXiv preprint arXiv:2209.04836.
