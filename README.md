# Neural network symmetries and linear mode connectivity

## Requirements

Install `open_lth` in same directory.

## Usage

To train models for analysis, run `train-models.sh`.

To run experiments, run `exp-run.sh`.

See `exp-plots.ipynb` for results.

Assumptions made by `nnperm`:
* BatchNorm precedes non-linearity
* if last Conv layer has X output channels, final linear layer takes X inputs
    - this means pooling/stride should reduce all image dims
* network is not going to be trained further (BatchNorm running mean/var aren't used)
* if using cache=True in get_normalizing_permutation, loss function is applied elementwise (e.g. this is true for MSE or MAE loss)

ResNet assumptions:
* Conv layer has no bias, but is followed by BatchNorm with bias
* shortcut connections always have weights (if not transformed, set as identity matrix without bias)
* the first shortcut points to the output of the first layer, subsequent shortcuts point to output of previous shortcut
* shortcuts apply an optional linear transform, then are added to the output of the previous (block) layer

## Development

Permutations and scaling in `nnperm.py`.
Permutation finding algorithm using optimal transport in `sinkhorn.py`.
Original implementation of geometric realignment by Udbhav Bamba (with modifications) kept in `nnperm_old.py` for testing purposes. 

To run unit tests, call `python test_nnperm.py`.
