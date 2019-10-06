# VItamin
:star: Star us on GitHub — it helps!

VItamin is a LIGO tool for predicting parameter 
posteriors given a gravitational wave time series. 
It produces training/testing sets, trains on those 
sets and compares its predictions to those 
from the bilby Bayesian inference library.

## Table of contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Links](#links)

## Installation

The prefered method of installation is via the 
anaconda package manager. An alternative method 
using pip and virtualenv may also be used. Instructions 
using this alternative method will also be given 
below. 

### Anaconda Installation Option

Create a virtual environment using 
the anaconda package manager. 

`conda update conda`

`conda create -n myenv python=3.6 anaconda`

Source your environment

`source activate myenv`

Install required packages. Anaconda will also 
handle all non-python packages needed.

`conda install requirements.txt`

### Alternative Installation Option

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

## Usage

This section details instructions for performing your 
own tests on sets of gravitational wave signals. Generation of 
test sets will require access to a large CPU cluster.

### Making Training sets

In order to generate your own training sets, you will need 
to edit the file `VICI_code_usage_example.py`. Within 
`VICI_code_usage_example.py` there will be a function 
at the top of the script titled `def get_params()`. Make 
sure that the following hyperparameters are set the values 
listed below.

```
load_train_set = False
train_set_dir = 'location/to/put/trainingset/data'
do_only_test = False
```

Now to generate your training sets, simply run 
the following command (attention, this will also 
initiate training after training set has been 
generated).

`python VICI_code_usage_example.py`

### Making Testing sets

Test sets are generated currently using the 
a large computing cluster at the LIGO Caltech site. 
Make sure that you are first logged into a condor-enabled 
computing cluster. Once on the cluster `cd` into the following 
directory of this repository.

`cd condor_runs/base_scripts`

Run the python script `make_dag.py`. This will generate a file 
titled `my.dag`. You can choose the number of test samples 
to generate by setting the `r` variable in the function 
`def main` to the number of test samples desired squared 
(e.g. r = 5 will make 25 test samples).

To generate test sample posteriors, submit your dag file 
to the condor computing cluster by runnin the following command.

`condor_submit_dag my.dag`

This will run the bilby_pe.py script for each test sample. 
You may change the hyperparameters for each of the samplers 
used by editing the bilby_pe.py script.

## License

VItamin is licensed under the terms of the MIT Open Source
license and is available for free.

## Links
* [arXiv paper](https://arxiv.org/abs/1909.06296)
