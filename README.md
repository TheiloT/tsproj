# Convolutional Dictionary Learning with Grid Refinement for Spike Sorting

Mini-Project for the MVA class ML for Timeseries, based on the study of the article:

> Convolutional sparse coding (CSC) and convolutional dictionary learning (CDL) for off-the-grid events.
> Song, A., Flores, F., and Ba D., **Convolutional Dictionary Learning with Grid Refinement**, *IEEE Transaction on Signal Processing*, 2020

Report is available [here](https://github.com/florian6973/tsproj/blob/main/31_Report.pdf)

The parts of the code coming from [srcdl](https://github.com/ds2p/srcdl) are highlighted with comments.

## Installation

Please create a specific Python 3.11 environment, and then install the cdlgr package: `pip install -e .`. 
You will then be able to run the command `ssrun` from the package.

Note that you may need to set up Kachery Cloud (*i.e.* link it with your Github account) to be able to download spikeforest datasets, with the command `kachery-cloud-init` (if automatic configuration fails).

## Use

Please follow the notebook `cdlgr.ipynb` to see how to run the code and view the results.

It has been tested in VS Code.