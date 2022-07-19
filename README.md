# Neural network symmetries and linear mode connectivity

## Requirements

Install `open_lth` in same directory.

## Usage

To train models for analysis, run `train-models.sh`.

To run experiments, run `exp-run.sh`.

See `exp-plots.ipynb` for results.

Assumptions made by `nnperm`:
* BatchNorm precedes non-linearity
* for ResNets, Conv layer has no bias, but is followed by BatchNorm with bias
* if last Conv layer has X output channels, final linear layer takes X inputs
    - this means pooling/stride should reduce all image dims
* network is not going to be trained further (BatchNorm running mean/var aren't used)

## Development

Geometric realignment algorithm and supporting functions in `nnperm.py`.
Previous implementation (with modifications) is kept in `nnperm_old.py` for testing purposes. 

To run unit tests, call `python test_nnperm.py`.
