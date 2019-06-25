#######################################################################################################################

# -- Example Code for using the Variational Inference for Computational Imaging (VICI) Model --

#######################################################################################################################

import numpy as np
import tensorflow as tf
import scipy.io as sio
import scipy.misc as mis
import h5py
from sys import exit

from Models import VICI_forward_model
from Models import VICI_inverse_model
from Models import CVAE
from Neural_Networks import batch_manager
#from Observation_Models import simulate_observations
from data import make_samples, chris_data
import plots

run_label='gpu2',            # label for run
plot_dir="/home/hunter.gabbard/public_html/CBC/cINNamon/gausian_results/VICI/%s" % run_label,                 # plot directory
ndata=512                    # y dimension size
load_mcmc_samp = True      # if True, load previously made train/test samples (including mcmc run).

# Defining the list of parameter that need to be fed into the models
def get_params():
    params = dict(
        image_size = [1,ndata], # Images Size
        print_values=True, # optionally print values every report interval
        n_samples = 5000, # number of posterior samples to save per reconstruction upon inference 
        num_iterations=300001, # number of iterations inference model (inverse reconstruction)
        initial_training_rate=0.00001, # initial training rate for ADAM optimiser inference model (inverse reconstruction)
        batch_size=1000, # batch size inference model (inverse reconstruction)
        report_interval=500, # interval at which to save objective function values and optionally print info during inference training
        z_dimension=50, # number of latent space dimensions inference model (inverse reconstruction)
        n_weights = 1024, # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        save_interval=1000, # interval at which to save inference model weights
        num_iterations_fw= 20001, # number of iterations of multifidelity forward model training
        initial_training_rate_fw=0.00002, # initial training rate for ADAM optimiser of multifidelity forward model training
        report_interval_fw=500, # interval at which to save objective function values and optionally print info during multifidelity forward model training
        z_dimensions_fw = 5, # latent space dimensionality of forward model
        n_weights_fw = 128, # intermediate layers dimensionality in forward model neural networks
        batch_size_fw=1000, # batch size of multifidelity forward model training
        save_interval_fw=2000, # interval at which to save multi-fidelity forward model weights

        ndata = ndata,
        r = 5,                      # the grid dimension for the output tests
        ndim_x=3,                    # number of parameters to PE on
        sigma=0.2,                   # stadnard deviation of the noise on signal
        usepars=[0,1,2],             # which parameters you want to do PE on
        tot_dataset_size=int(2**20), # total size of training set
        seed=42,                     # random seed number
        run_label=run_label,            # label for run
        plot_dir=plot_dir,                 # plot directory
        parnames=['A','t0','tau','phi','w'],    # parameter names
        n_burnin=2000                # number of burn-in samples to use in MCMC chains
    )
    return params 

# You will need two types of sets to train the model: 
#
# 1) High-Fidelity Set. Small set, but with accurate measurements paired to the groundtruths. Will need the following sets:
#    - x_data_train_h = Ground-truths for which you have accurate mesurements (point estimates on pars)
#    - y_data_train_h = Accurate mesurements corresponding to x_data_train_h (noisy waveforms)
#    - y_data_train_lh = Inaccurate mesurements corresponding to x_data_train_h (noise-free waveforms)
#
# 2) Low-Fidelity Set. Large set, but without accurate measurements paired to the groundtruths. Will need the following sets:
#    - x_data_train = Ground-truths for which you only have inaccurate mesurements (point estimates on pars) 
#    - y_data_train_l = Inaccurate mesurements corresponding to x_data_train (noise-free waveforms)
#
# To run the model once it is trained you will need:
#
# y_data_test_h - new measurements you want to infer a solution posterior from    
#
# All inputs and outputs are in the form of 2D arrays, where different objects are along dimension 0 and elements of the same object are along dimension 1


# Get the training/test data and parameters of run
params=get_params()


# Get mcmc samples
if load_mcmc_samp:
    hf = h5py.File('data/generated_samples_%s' % run_label, 'r')
    x_data_train_h = np.array(hf.get('x_data_train_h'))
    y_data_train_lh = np.array(hf.get('y_data_train_lh'))
    y_data_test_h = np.array(hf.get('y_data_test_h') )
    pos_test = np.array(hf.get('pos_test'))
    samples = np.array(hf.get('samples'))
    x_data_train, y_data_train_l, y_data_train_h = x_data_train_h, y_data_train_lh, y_data_train_lh
    hf.close()
else:
    x_data_train_h, _, y_data_train_lh, y_data_test_h,pos_test = make_samples.get_sets(params)
    x_data_train, y_data_train_l, y_data_train_h = x_data_train_h, y_data_train_lh, y_data_train_lh
    samples = chris_data.mcmc_sampler(params['r'],params['n_samples'],params['ndim_x'],y_data_test_h,params['sigma'],params['usepars'],params['n_burnin'])

    # save samples for latter
    f = h5py.File("data/generated_samples_%s" % run_label, "w")
    f.create_dataset("x_data_train_h", data=x_data_train_h)
    f.create_dataset("y_data_train_lh", data=y_data_train_lh)
    f.create_dataset("y_data_test_h", data=y_data_test_h)
    f.create_dataset("pos_test", data=pos_test)
    f.create_dataset("samples", data=samples)
    f.close()

# Make directory for plots
#plots.make_dirs(params['plot_dir'][0])
# Declare plot class variables
plotter = plots.make_plots(params,samples,None,pos_test)

# First, we learn a multi-fidelity model that lerns to infer high-fidelity (accurate) observations from trget images/objects and low fidelity simulated observations. for this we use the portion of the training set for which we do have real/high fidelity observations.
#_, _ = VICI_forward_model.train(params, x_data_train_h, y_data_train_h, y_data_train_lh, "forward_model_dir/forward_model.ckpt", plotter) # This trains the forward model and saves the weights in forward_model_dir/forward_model.ckpt

# We then train the inference model using all training images and associated low-fidelity (inaccurate) observations. Using the previously trained forward model to draw from the observation likelihood.
#_, _ = VICI_inverse_model.train(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "forward_model_dir/forward_model.ckpt", "inverse_model_dir/inverse_model.ckpt", plotter, y_data_test_h) # This trains the inverse model to recover posteriors using the forward model weights stored in forward_model_dir/forward_model.ckpt and saves the inverse model weights in inverse_model_dir/inverse_model.ckpt 
_, _ = VICI_inverse_model.resume_training(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "forward_model_dir/forward_model.ckpt", "inverse_model_dir/inverse_model.ckpt")

# The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
xm, xsx, XS, pmax = VICI_inverse_model.run(params, y_data_test_h, np.shape(x_data_train)[1], "inverse_model_dir/inverse_model.ckpt") # This runs the trained model using the weights stored in inverse_model_dir/inverse_model.ckpt
# The outputs are the following:
# - xm = marginal means
# - xsx = marginal standard deviations
# - XS = draws from the posterior (3D array with different samples for the same input along the third dimension)
# - pmax = approximate maxima (approximate 'best' reconstructions)

# Generate final results plots
plotter = plots.make_plots(params,samples,XS,pos_test)

# Geneerate overlap scatter plots
plotter.make_overlap_plot()
