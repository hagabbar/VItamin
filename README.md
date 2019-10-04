# VItamin
:star: Star us on GitHub — it helps!

VItamin is a LIGO tool for predicting parameter 
posteriors given a gravitational wave time series. 
It produces training/testing sets, trains on those 
sets and compares its predictions to those 
from the bilby Bayesian inference library.

## Table of contents
- [Installation](#installation)
- [License](#license)
- [Links](#links)

## Installation

The prefered method of installation is via the 
anaconda package manager. An alternative method 
using pip and virtualenv may also be used. Instructions 
using this alternative method will also be given 
below. 

### Anaconda Option

Create a virtual environment using 
the anaconda package manager. 

`conda update conda`

`conda create -n yourenvname python=x.x anaconda`

Source your environmet

`source activate yourenvname`

Install required packages. Anaconda will also 
handle all non-python packages needed.

`conda install requirements.txt`

### Alternative Option

First, ensure that you have both CUDA and CUDNN 
installed on your machine. This is required 
in order to run tensorflow on a GPU (which 
will speed up training).

Create a virtual 
environment where the required dependency packages 
will be installed.

`virtualenv -p python3.6 myenv`

Source the virtual environment

`source myenv`

Install the required packages via pip.

`pip install requirements.txt`


## License

VItamin is licensed under the terms of the MIT Open Source
license and is available for free.

## Links
* [arXiv paper](https://arxiv.org/abs/1909.06296)
