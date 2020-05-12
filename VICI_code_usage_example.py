######################################################################################################################

# -- Variational Inference for Gravitational wave Parameter Estimation --


#######################################################################################################################

import argparse
import numpy as np
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
import scipy.io as sio
import scipy.misc as mis
import h5py
from sys import exit
import shutil
import os
import bilby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from time import strftime
import corner
import glob

from Models import VICI_inverse_model
from Models import CVAE
from bilby_pe import run
from Neural_Networks import batch_manager
from data import chris_data
import plots

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--gen_train", default=False, help="generate the training data")
parser.add_argument("--gen_test", default=False, help="generate the testing data")
parser.add_argument("--train", default=False, help="train the network")
parser.add_argument("--test", default=False, help="test the network")
args = parser.parse_args()

# fixed parameter values
fixed_vals = {'mass_1':50.0,
        'mass_2':50.0,
        'mc':None,
        'geocent_time':0.0,
        'phase':0.0,
        'ra':1.375,
        'dec':-1.2108,
        'psi':0.0,
        'theta_jn':0.0,
        'luminosity_distance':2000.0,
        'a_1':0.0,
        'a_2':0.0,
	'tilt_1':0.0,
	'tilt_2':0.0,
        'phi_12':0.0,
        'phi_jl':0.0,
        'det':['H1','L1','V1']}

# prior bounds
bounds = {'mass_1_min':35.0, 'mass_1_max':80.0,
        'mass_2_min':35.0, 'mass_2_max':80.0,
        'M_min':70.0, 'M_max':160.0,
        'geocent_time_min':0.15,'geocent_time_max':0.35,
        'phase_min':0.0, 'phase_max':2.0*np.pi,
        'ra_min':0.0, 'ra_max':2.0*np.pi,
        'dec_min':-0.5*np.pi, 'dec_max':0.5*np.pi,
        'psi_min':0.0, 'psi_max':2.0*np.pi,
        'theta_jn_min':0.0, 'theta_jn_max':np.pi,
        'a_1_min':0.0, 'a_1_max':0.0,
        'a_2_min':0.0, 'a_2_max':0.0,
        'tilt_1_min':0.0, 'tilt_1_max':0.0,
        'tilt_2_min':0.0, 'tilt_2_max':0.0,
        'phi_12_min':0.0, 'phi_12_max':0.0,
        'phi_jl_min':0.0, 'phi_jl_max':0.0,
        'luminosity_distance_min':1000.0, 'luminosity_distance_max':3000.0}

# define which gpu to use during training
gpu_num = str(5)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

n_fc = 2048

# Defining the list of parameter that need to be fed into the models
def get_params():

    ndata = 256 # length of input to NN == fs * num_detectors
    rand_pars = ['mass_1','mass_2','luminosity_distance','geocent_time','phase','theta_jn','psi','ra','dec']
    run_label = 'multi-modal_%ddet_%dpar_%dHz_run49' % (len(fixed_vals['det']),len(rand_pars),ndata)
    bilby_results_label = 'attempt_to_fix_astropy_bug'
    r = 6                       # number of test samples to use for plotting
    pe_test_num = 256               # total number of test samples available to use in directory
    tot_dataset_size = int(1e7)    # total number of training samples to use

    tset_split = int(1e3)          # number of training samples per saved data files
    save_interval = int(1e5)
    ref_geocent_time=1126259642.5   # reference gps time
    load_chunk_size = 1e6
    batch_size = 64
    params = dict(
        gpu_num=gpu_num,
        resume_training=False,           # if True, resume training from checkpoint
        ndata = ndata,
        run_label=run_label,            # label for run
        bilby_results_label=bilby_results_label, # label given to results for bilby posteriors
        tot_dataset_size = tot_dataset_size,
        tset_split = tset_split, 
        plot_dir="/home/hunter.gabbard/public_html/CBC/VItamin/gw_results/%s" % run_label,                 # plot directory
        hyperparam_optim = False,      # optimize hyperparameters for model 
        hyperparam_optim_stop = int(2e5), # stopping point of hyperparameter optimizer 
        hyperparam_n_call = 30,       # number of optimization calls
        load_by_chunks = True,
        load_chunk_size = load_chunk_size,
        load_iteration = int((load_chunk_size * 25)/batch_size),
        weight_init = 'xavier',#[xavier,VarianceScaling,Orthogonal]
        ramp = True,                 # if true, do ramp on KL loss
        KL_coef = 1e0,                # coefficient to place in front of KL loss

        print_values=True,            # optionally print values every report interval
        n_samples = 3000,             # number of posterior samples to save per reconstruction upon inference 
        num_iterations=int(1e7)+1,    # number of iterations inference model (inverse reconstruction)
        initial_training_rate=1e-4, # initial training rate for ADAM optimiser inference model (inverse reconstruction)
        batch_size=batch_size,               # batch size inference model (inverse reconstruction)
        batch_norm=True,              # if true, do batch normalization in all layers
        l1_loss = False,               # apply l1 regularization on mode weights
        report_interval=500,          # interval at which to save objective function values and optionally print info during inference training
               # number of latent space dimensions inference model (inverse reconstruction)
        n_modes=7,                  # number of modes in the latent space
        n_hlayers=3,                # the number of hidden layers in each network
        n_convsteps = 0,              # Set to zero if not wanted. the number of convolutional steps used to prepare the y data (size changes by factor of  n_filter/(2**n_redsteps) )
        reduce = False,
        n_conv = 4,                # number of convolutional layers to use in each part of the networks. None if not used
        by_channel = True,        # if True, do convolutions as seperate channels
        n_filters = [33, 33, 33, 33],#,16,32,32],
        filter_size = [3, 3, 3, 3],#,3,3,3],
        drate = 0.5,
        maxpool = [1,2,1,1],#,	2,1,2],
        conv_strides = [1,1,1,1],#,1,1,1],
        pool_strides = [1,2,1,1],#,2,1,2],
        ramp_start = 1e4,
        ramp_end = 1e5,
        save_interval=save_interval,           # interval at which to save inference model weights
        plot_interval=save_interval,           # interval over which plotting is done
        z_dimension=100,                    # number of latent space dimensions inference model (inverse reconstruction)
        n_weights_r1 = [n_fc,n_fc,n_fc],             # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        n_weights_r2 = [n_fc,n_fc,n_fc],             # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        n_weights_q = [n_fc,n_fc,n_fc],              # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        duration = 1.0,                             # the timeseries length in seconds
        r = r,                                      # the grid dimension for the output tests
        rand_pars=rand_pars,
        corner_parnames = ['m_{1}\,(\mathrm{M}_{\odot})','m_{2}\,(\mathrm{M}_{\odot})','d_{\mathrm{L}}\,(\mathrm{Mpc})','t_{0}\,(\mathrm{seconds})','{\phi}','\Theta_{jn}','{\psi}','\mathrm{RA}','\mathrm{DEC}'],
        cornercorner_parnames = ['$m_{1}\,(\mathrm{M}_{\odot})$','$m_{2}\,(\mathrm{M}_{\odot})$','$d_{\mathrm{L}}\,(\mathrm{Mpc})$','$t_{0}\,(\mathrm{seconds})$','${\phi}$','$\Theta_{jn}$','${\psi}$','$\mathrm{RA}$','$\mathrm{DEC}$'],
        ref_geocent_time=ref_geocent_time,            # reference gps time
        training_data_seed=43,                              # random seed number
        testing_data_seed=44,
        wrap_pars=[],#['phase','psi'],                  # parameters that get wrapped on the 1D parameter 
        weighted_pars=['ra','dec','geocent_time'],#['ra','dec','goecent_time','theta_jn'],                     # set to None if not using, pars to weight during training
        weighted_pars_factor=1,                         # weighting scalar factor
        inf_pars=['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','ra','dec'],
        train_set_dir='/home/hunter.gabbard/CBC/VItamin/training_sets_second_sub_%ddet_%dpar_%dHz/tset_tot-%d_split-%d' % (len(fixed_vals['det']),len(rand_pars),ndata,tot_dataset_size,tset_split), #location of training set
#        test_set_dir='/home/hunter.gabbard/CBC/VItamin/condor_runs_second_paper_sub/%dpar_%dHz_%ddet_case_%dtest/test_waveforms' % (len(rand_pars),ndata,len(fixed_vals['det']),pe_test_num), #location of test set
#        pe_dir='/home/hunter.gabbard/CBC/VItamin/condor_runs_second_paper_sub/%dpar_%dHz_%ddet_case_%dtest/test' % (len(rand_pars),ndata,len(fixed_vals['det']),pe_test_num),    # location of bilby PE results
        test_set_dir='/home/hunter.gabbard/CBC/VItamin/condor_runs_second_paper_sub/attempt_to_fix_astropy_bug/test_waveforms',
        pe_dir='/home/hunter.gabbard/CBC/VItamin/condor_runs_second_paper_sub/attempt_to_fix_astropy_bug/test',
        KL_cycles = 1,                                                         # number of cycles to repeat for the KL approximation
        load_plot_data=False,                                                  # use old plotting data
        samplers=['vitamin','dynesty','cpnest'],#,'ptemcee','emcee'],          # samplers to use when plotting

        #add_noise_real=True,                  # whether or not to add extra noise realizations in training set
        #do_normscale=True,                    # if true normalize parameters
        #do_mc_eta_conversion=False,           # if True, convert m1 and m2 parameters into mc and eta
        #n_kl_samp=int(r*r),                        # number of iterations in statistic tests TODO: remove this
        #do_adkskl_test=False,                  # if True, do statistic tests
        #do_m1_m2_cut=False,                   # if True, make a cut on all m1 and m2 values    
        #do_extra_noise=True,                  # add extra noise realizations during training
        #Npp = int(r*r),                             # number of test signals per pp-plot. TODO: use same 
                                              # use same samples as bilby
        #use_samplers = [0,1],                  # number of Bilby samplers to use 
        #kl_set_dir='/home/chrism/kl_output', # location of test set used for kl
        #do_only_test = False,                  # if true, don't train but only run on test samples using pretrained network
        #load_plot_data = False,                # if true, load in previously generated plot data, otherwise generate plots from scratch
        doPE = True,                          # if True then do bilby PE
        #whitening_factor = np.sqrt(float(ndata)) # whitening scale factor
    )
    return params

# Get training/test data and parameters of run
params=get_params()
f = open("params_%s.txt" % params['run_label'],"w")
f.write( str(params) )
f.close()

kernel_1 = Integer(low=3, high=9, name='kernel_1')
strides_1 = Integer(low=1, high=2, name='strides_1')
pool_1 = Integer(low=1, high=2, name='pool_1')
kernel_2 = Integer(low=3, high=9, name='kernel_2')
strides_2 = Integer(low=1, high=2, name='strides_2')
pool_2 = Integer(low=1, high=2, name='pool_2')
kernel_3 = Integer(low=3, high=9, name='kernel_3')
strides_3 = Integer(low=1, high=2, name='strides_3')
pool_3 = Integer(low=1, high=2, name='pool_3')
#kernel_4 = Integer(low=3, high=9, name='kernel_4')
#strides_4 = Integer(low=1, high=2, name='strides_4')
#pool_4 = Integer(low=1, high=2, name='pool_4')
#kernel_5 = Integer(low=3, high=9, name='kernel_5')
#strides_5 = Integer(low=1, high=2, name='strides_5')
#pool_5 = Integer(low=1, high=2, name='pool_5')
z_dimension = Integer(low=8, high=500, name='z_dimension')
n_modes = Integer(low=8, high=100, name='n_modes')
n_filters_1 = Integer(low=4, high=64, name='n_filters_1')
n_filters_2 = Integer(low=4, high=64, name='n_filters_2')
n_filters_3 = Integer(low=4, high=64, name='n_filters_3')
batch_size = Integer(low=params['batch_size']-1, high=params['batch_size'], name='batch_size')
n_weights_fc_1 = Integer(low=50, high=2048, name='n_weights_fc_1')
n_weights_fc_2 = Integer(low=50, high=2048, name='n_weights_fc_2')
n_weights_fc_3 = Integer(low=50, high=2048, name='n_weights_fc_3')
#n_weights_r2_1 = Integer(low=1, high=2048, name='n_weights_r2_1')
#n_weights_r2_2 = Integer(low=1, high=2048, name='n_weights_r2_2')
#n_weights_r2_3 = Integer(low=1, high=2048, name='n_weights_r2_3')
#n_weights_q_1 = Integer(low=1, high=2048, name='n_weights_q_1')
#n_weights_q_2 = Integer(low=1, high=2048, name='n_weights_q_2')
#n_weights_q_3 = Integer(low=1, high=2048, name='n_weights_q_3')

dimensions = [kernel_1, 
              strides_1,
              pool_1,
              kernel_2, 
              strides_2,
              pool_2,
              kernel_3,
              strides_3,
              pool_3,
              z_dimension,
              n_modes,
              n_filters_1,
              n_filters_2,
              n_filters_3,
              batch_size,
              n_weights_fc_1,
              n_weights_fc_2,
              n_weights_fc_3]
#              n_weights_r2_1,
#              n_weights_r2_2,
#              n_weights_r2_3,
#              n_weights_q_1,
#              n_weights_q_2,
#              n_weights_q_3]
#              kernel_4,
#              strides_4,
#              pool_4,
#              kernel_5,
#              strides_5,
#              pool_5]

"""
default_hyperparams = [params['filter_size'][0],
                       params['conv_strides'][0],
                       params['maxpool'][0],
                       params['filter_size'][1],
                       params['conv_strides'][1],
                       params['maxpool'][1],
                       params['filter_size'][2],
                       params['conv_strides'][2],
                       params['maxpool'][2],
                       params['z_dimension'],
                       params['n_modes'],
                       params['n_filters'][0],
                       params['n_filters'][1],
                       params['n_filters'][2],
                       params['batch_size'],
                       params['n_weights_r1'][0],
                       params['n_weights_r1'][1],
                       params['n_weights_r1'][2],
#                       params['n_weights_r2'][0],
#                       params['n_weights_r2'][1],
#                       params['n_weights_r2'][2],
#                       params['n_weights_q'][0],
#                       params['n_weights_q'][1],
#                       params['n_weights_q'][2]
                      ]
"""
best_loss = 1000

def load_data(input_dir,inf_pars,load_condor=False):
#    tf.compat.v1.enable_eager_execution() 

    # load generated samples back in
    train_files = []
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': []}

    if load_condor == True:
        filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    else:
        filenames = os.listdir(dataLocations[0])

    snrs = []
    for filename in filenames:
        try:
            train_files.append(filename)

        except OSError:
            print('Could not load requested file')
            continue

    if params['load_by_chunks'] == True and load_condor == False:
        train_files_idx = np.arange(len(train_files))[:int(params['load_chunk_size']/1000.0)]
        np.random.shuffle(train_files_idx)
        train_files = np.array(train_files)[train_files_idx]

    for filename in train_files:
        try:
            print(filename)
            data_temp={'x_data': h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data'][:],
                  'y_data_noisefree': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisefree'][:],
                  'y_data_noisy': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisy'][:],
                  'rand_pars': h5py.File(dataLocations[0]+'/'+filename, 'r')['rand_pars'][:]}
            data['x_data'].append(data_temp['x_data'])
            data['y_data_noisefree'].append(np.expand_dims(data_temp['y_data_noisefree'], axis=0))
            data['y_data_noisy'].append(np.expand_dims(data_temp['y_data_noisy'], axis=0))
            data['rand_pars'] = data_temp['rand_pars']
            snrs.append(h5py.File(dataLocations[0]+'/'+filename, 'r')['snrs'][:])
        except OSError:
            print('Could not load requested file')
            continue

    snrs = np.array(snrs)
#    plt.hist(snrs[0,:,0])
#    plt.hist(snrs[0,:,1])
#    plt.hist(snrs[0,:,2])
#    plt.savefig('/home/hunter.gabbard/public_html/test.png')
#    plt.close()
#    exit()
    # extract the prior bounds
    bounds = {}
    for k in data_temp['rand_pars']:
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'
        bounds[par_max] = h5py.File(dataLocations[0]+'/'+filename, 'r')[par_max][...].item()
        bounds[par_min] = h5py.File(dataLocations[0]+'/'+filename, 'r')[par_min][...].item()
    data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
    data['y_data_noisefree'] = np.concatenate(np.array(data['y_data_noisefree']), axis=0)
    data['y_data_noisy'] = np.concatenate(np.array(data['y_data_noisy']), axis=0)
    

    # normalise the data parameters
    for i,k in enumerate(data_temp['rand_pars']):
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'

        data['x_data'][:,i]=(data['x_data'][:,i] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
    x_data = data['x_data']
    y_data = data['y_data_noisefree']
    y_data_noisy = data['y_data_noisy']
    if params['load_by_chunks'] == True:
        y_normscale = 36.43879218007172 
    else:
        y_normscale = np.max(np.abs(y_data_noisy))
    
    # extract inference parameters
    idx = []
    for k in inf_pars:
        print(k)
        for i,q in enumerate(data['rand_pars']):
            m = q.decode('utf-8')
            if k==m:
                idx.append(i)
    x_data = x_data[:,idx]


    return x_data, y_data, y_data_noisy, y_normscale, snrs

@use_named_args(dimensions=dimensions)
def hyperparam_fitness(kernel_1, strides_1, pool_1,
                       kernel_2, strides_2, pool_2,
                       kernel_3, strides_3, pool_3,
                       z_dimension,n_modes,
                       n_filters_1,n_filters_2,n_filters_3,
                       batch_size,
                       n_weights_fc_1,n_weights_fc_2,n_weights_fc_3):
#                       n_weights_r2_1,n_weights_r2_2,n_weights_r2_3,
#                       n_weights_q_1,n_weights_q_2,n_weights_q_3):

    # set tunable hyper-parameters
    params['filter_size'] = [kernel_1,kernel_2,kernel_3]
    params['n_filters'] = [n_filters_1,n_filters_2,n_filters_3]
    for filt_idx in range(len(params['n_filters'])):
        if (params['n_filters'][filt_idx] % 3) != 0:
            # keep adding 1 until filter size is divisible by 3
            while (params['n_filters'][filt_idx] % 3) != 0:
                params['n_filters'][filt_idx] += 1
    params['conv_strides'] = [strides_1,strides_2,strides_3]
    params['maxpool'] = [pool_1,pool_2,pool_3]
    params['pool_strides'] = [pool_1,pool_2,pool_3]
    params['z_dimension'] = z_dimension
    params['n_modes'] = n_modes
    params['batch_size'] = batch_size
    params['n_weights_r1'] = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3]
    params['n_weights_r2'] = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3]
    params['n_weights_q'] = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3]

    # print hyper-parameters
    # Print the hyper-parameters.
    print('kernel_1: {}'.format(kernel_1))
    print('strides_1: {}'.format(strides_1))
    print('pool_1: {}'.format(pool_1))
    print('kernel_2: {}'.format(kernel_2))
    print('strides_2: {}'.format(strides_2))
    print('pool_2: {}'.format(pool_2))
    print('kernel_3: {}'.format(kernel_3))
    print('strides_3: {}'.format(strides_3))
    print('pool_3: {}'.format(pool_3))
#    print('kernel_4: {}'.format(kernel_4))
#    print('strides_4: {}'.format(strides_4))
#    print('pool_4: {}'.format(pool_4))
#    print('kernel_5: {}'.format(kernel_5))
#    print('strides_5: {}'.format(strides_5))
#    print('pool_5: {}'.format(pool_5))
    print('z_dimension: {}'.format(z_dimension))
    print('n_modes: {}'.format(n_modes))
    print('n_filters_1: {}'.format(params['n_filters'][0]))
    print('n_filters_2: {}'.format(params['n_filters'][1]))
    print('n_filters_3: {}'.format(params['n_filters'][2]))
    print('batch_size: {}'.format(batch_size))
    print('n_weights_r1_1: {}'.format(n_weights_fc_1))
    print('n_weights_r1_2: {}'.format(n_weights_fc_2))
    print('n_weights_r1_3: {}'.format(n_weights_fc_3))
#    print('n_weights_r2_1: {}'.format(n_weights_r2_1))
#    print('n_weights_r2_2: {}'.format(n_weights_r2_2))
#    print('n_weights_r2_3: {}'.format(n_weights_r2_3))
#    print('n_weights_q_1: {}'.format(n_weights_q_1))
#    print('n_weights_q_2: {}'.format(n_weights_q_2))
#    print('n_weights_q_3: {}'.format(n_weights_q_3))
    print()

    start_time = time.time()
    print('start time: {}'.format(strftime('%X %x %Z'))) 
    # Train model with given hyperparameters
    VICI_loss, VICI_session, VICI_saver, VICI_savedir = VICI_inverse_model.train(params, x_data_train, y_data_train,
                             x_data_test, y_data_test, y_data_test_noisefree,
                             y_normscale,
                             "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'],
                             x_data_test, bounds, fixed_vals,
                             XS_all)

    end_time = time.time()
    print('Run time : {} h'.format((end_time-start_time)/3600))

    # Print the loss.
    print()
    print("Total loss: {0:.2}".format(VICI_loss))
    print()

    # update variable outside of this function using global keyword
    global best_loss

    # save model if new best model
    if VICI_loss < best_loss:

        # Save model 
        save_path = VICI_saver.save(VICI_session,VICI_savedir)

        # save hyperparameters
        converged_hyperpar_dict = dict(filter_size = params['filter_size'],
                                       conv_strides = params['conv_strides'],
                                       maxpool = params['maxpool'],
                                       pool_strides = params['pool_strides'],
                                       z_dimension = params['z_dimension'],
                                       n_modes = params['n_modes'],
                                       n_filters = params['n_filters'],
                                       batch_size = params['batch_size'],
                                       n_weights_fc = params['n_weights_r1'],
                                       best_loss = best_loss)
                                       #n_weights_r2 = params['n_weights_r2'],
                                       #n_weights_q = params['n_weights_q'])

        f = open("inverse_model_dir_%s/converged_hyperparams.txt" % params['run_label'],"w")
        f.write( str(converged_hyperpar_dict) )
        f.close()

        # update the best loss
        best_loss = VICI_loss
        
        # Print the loss.
        print()
        print("New best loss: {0:.2}".format(best_loss))
        print()

    # clear tensorflow session
    VICI_session.close()

    return VICI_loss

#####################################################
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
#####################################################

# Make training samples
if args.gen_train:
    
    # Make training set directory
    os.system('mkdir -p %s' % params['train_set_dir'])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    # Iterate over number of requested training samples
    for i in range(0,params['tot_dataset_size'],params['tset_split']):

        _, signal_train, signal_train_pars,snrs = run(sampling_frequency=params['ndata']/params['duration'],
                                                          duration=params['duration'],
                                                          N_gen=params['tset_split'],
                                                          ref_geocent_time=params['ref_geocent_time'],
                                                          bounds=bounds,
                                                          fixed_vals=fixed_vals,
                                                          rand_pars=params['rand_pars'],
                                                          seed=params['training_data_seed']+i,
                                                          label=params['run_label'],
                                                          training=True)

        print("Generated: %s/data_%d-%d.h5py ..." % (params['train_set_dir'],(i+params['tset_split']),params['tot_dataset_size']))
        hf = h5py.File('%s/data_%d-%d.h5py' % (params['train_set_dir'],(i+params['tset_split']),params['tot_dataset_size']), 'w')
        for k, v in params.items():
            try:
                hf.create_dataset(k,data=v)
            except:
                pass
        hf.create_dataset('x_data', data=signal_train_pars)
        for k, v in bounds.items():
            hf.create_dataset(k,data=v)
        hf.create_dataset('y_data_noisy', data=signal_train)
        hf.create_dataset('y_data_noisefree', data=signal_train)
        hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
        hf.create_dataset('snrs', data=snrs)
        hf.close()

# Make testing set directory
if args.gen_test:

    # Make testing set directory
    os.system('mkdir -p %s' % params['test_set_dir'])

    # Maketesting samples
    signal_test_noisy, signal_test_noisefree, signal_test_pars = [], [], [] 
    for i in range(params['r']*params['r']):
        temp_noisy, temp_noisefree, temp_pars = run(sampling_frequency=params['ndata']/params['duration'],
                                                      duration=params['duration'],
                                                      N_gen=1,
                                                      ref_geocent_time=params['ref_geocent_time'],
                                                      bounds=bounds,
                                                      fixed_vals=fixed_vals,
                                                      rand_pars=params['rand_pars'],
                                                      inf_pars=params['inf_pars'],
                                                      label=params['run_label'] + '_' + str(i),
                                                      out_dir=params['pe_dir'],
                                                      training=False,
                                                      seed=params['testing_data_seed']+i,
                                                      do_pe=params['doPE'])
        signal_test_noisy.append(temp_noisy)
        signal_test_noisefree.append(temp_noisefree)
        signal_test_pars.append([temp_pars])

    print("Generated: %s/data_%s.h5py ..." % (params['test_set_dir'],params['run_label']))
    hf = h5py.File('%s/data_%d.h5py' % (params['test_set_dir'],params['r']*params['r']),'w')
    for k, v in params.items():
        try:
            hf.create_dataset(k,data=v)
        except:
            pass
    hf.create_dataset('x_data', data=signal_test_pars)
    for k, v in bounds.items():
        hf.create_dataset(k,data=v)
    hf.create_dataset('y_data_noisefree', data=signal_test_noisefree)
    hf.create_dataset('y_data_noisy', data=signal_test_noisy)
    hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
    hf.close()

# if we are now training the network
if args.train:

    # load the noisefree training data back in
    x_data_train, y_data_train, _, y_normscale, snrs_train = load_data(params['train_set_dir'],params['inf_pars'])

    # load the noisy testing data back in
    x_data_test, y_data_test_noisefree, y_data_test,_,snrs_test = load_data(params['test_set_dir'],params['inf_pars'],load_condor=True)

    # reshape arrays for multi-detector
    y_data_train = y_data_train.reshape(y_data_train.shape[0]*y_data_train.shape[1],y_data_train.shape[2]*y_data_train.shape[3])
#    y_data_train = y_data_train.reshape(y_data_train.shape[1],y_data_train.shape[2]*y_data_train.shape[3])
    y_data_test = y_data_test.reshape(y_data_test.shape[0],y_data_test.shape[1]*y_data_test.shape[2])
    y_data_test_noisefree = y_data_test_noisefree.reshape(y_data_test_noisefree.shape[0],y_data_test_noisefree.shape[1]*y_data_test_noisefree.shape[2])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    f = open('%s/latest_%s/params_%s.txt' % (params['plot_dir'],params['run_label'],params['run_label']),"w")
    f.write( str(params) )
    f.close()

    # load up the posterior samples (if they exist)
    # load generated samples back in
    post_files = []
    #~/bilby_outputs/bilby_output_dynesty1/multi-modal3_0.h5py

     # choose directory with lowest number of total finished posteriors
    num_finished_post = int(1e8)
    for i in params['samplers']:
        if i == 'vitamin':
            continue
        for j in range(1):
            input_dir = '%s_%s%d/' % (params['pe_dir'],i,j+1)
            if type("%s" % input_dir) is str:
                dataLocations = ["%s" % input_dir]

            filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))      
            if len(filenames) < num_finished_post:
                sampler_loc = i + str(j+1)

    dataLocations_try = '%s_%s' % (params['pe_dir'],sampler_loc)
    dataLocations = '%s_%s1' % (params['pe_dir'],params['samplers'][1])

    #for i,filename in enumerate(glob.glob(dataLocations[0])):
    i_idx = 0
    i = 0
    i_idx_use = []
    while i_idx < params['r']*params['r']:
#    for i in range(params['r']*params['r']):
        filename_try = '%s/%s_%d.h5py' % (dataLocations_try,params['bilby_results_label'],i)
        filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)

        # If file does not exist, skip to next file
        try:
            h5py.File(filename_try, 'r')
        except Exception as e:
            i+=1
            print(e)
            continue

        print(filename)
        post_files.append(filename)
        data_temp = {} 
        #bounds = {}
        n = 0
        for q in params['inf_pars']:
             p = q + '_post'
             par_min = q + '_min'
             par_max = q + '_max'
             data_temp[p] = h5py.File(filename, 'r')[p][:]
             if p == 'geocent_time_post':
                 data_temp[p] = data_temp[p] - params['ref_geocent_time']
#                 Nsamp = data_temp[p].shape[0]
#                 n = n + 1
#                 continue
             data_temp[p] = (data_temp[p] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
             Nsamp = data_temp[p].shape[0]
             n = n + 1
        XS = np.zeros((Nsamp,n))
        j = 0
        for p,d in data_temp.items():
            XS[:,j] = d
            j += 1

        # Make corner plot of VItamin posterior samples
#        figure = corner.corner(XS, labels=params['inf_pars'],
#                       quantiles=[0.16, 0.5, 0.84],
                       #range=[[0.0,1.0]]*np.shape(x_data_test)[1],
#                       truths=x_data_test[i,:],
#                       show_titles=True, title_kwargs={"fontsize": 12})

        if i_idx == 0:
            XS_all = np.expand_dims(XS[:params['n_samples'],:], axis=0)
        else:
            # save all posteriors in array
            XS_all = np.vstack((XS_all,np.expand_dims(XS[:params['n_samples'],:], axis=0)))


        for q_idx,q in enumerate(params['inf_pars']):
            par_min = q + '_min'
            par_max = q + '_max'

            #x_data_test[i,q_idx] = (x_data_test[i,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]

#        plt.savefig('%s/latest_%s/truepost_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))
        i_idx_use.append(i_idx)
        i+=1
        i_idx+=1


    y_data_test = y_data_test[i_idx_use,:]
    y_data_test_noisefree = y_data_test_noisefree[i_idx_use,:]
    x_data_test = x_data_test[i_idx_use,:]

    # reshape y data into channels last format for convolutional approach
    if params['reduce'] == True or params['n_conv'] != None:
        y_data_test_copy = np.zeros((y_data_test.shape[0],params['ndata'],len(fixed_vals['det'])))
        y_data_test_noisefree_copy = np.zeros((y_data_test_noisefree.shape[0],params['ndata'],len(fixed_vals['det'])))
        y_data_train_copy = np.zeros((y_data_train.shape[0],params['ndata'],len(fixed_vals['det'])))
        for i in range(y_data_test.shape[0]):
            for j in range(len(fixed_vals['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_test_copy[i,:,j] = y_data_test[i,idx_range]
                y_data_test_noisefree_copy[i,:,j] = y_data_test_noisefree[i,idx_range]
        y_data_test = y_data_test_copy
        y_data_noisefree_test = y_data_test_noisefree_copy

        for i in range(y_data_train.shape[0]):
            for j in range(len(fixed_vals['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_train_copy[i,:,j] = y_data_train[i,idx_range]
        y_data_train = y_data_train_copy

    # run hyperparameter optimization
    if params['hyperparam_optim'] == True:

        #hyperparam_fitness(x=default_hyperparams)

        # Run optimization
        search_result = gp_minimize(func=hyperparam_fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Negative Expected Improvement.
                            n_calls=params['hyperparam_n_call'],
                            x0=default_hyperparams)

        from skopt import dump
        dump(search_result, 'search_result_store')

        plot_convergence(search_result)
        plt.savefig('%s/latest_%s/hyperpar_convergence.png' % (params['plot_dir'],params['run_label']))
        print('Did a hyperparameter search') 

    # train using user defined params
    else:
        VICI_inverse_model.train(params, x_data_train, y_data_train,
                                 x_data_test, y_data_test, y_data_test_noisefree,
                                 y_normscale,
                                 "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'],
                                 x_data_test, bounds, fixed_vals,
                                 XS_all,snrs_test) 

# if we are now testing the network
if args.test:

    # load the noisefree training data back in
    x_data_train, y_data_train, _, y_normscale,snrs_train = load_data(params['train_set_dir'],params['inf_pars'])

#    print('YOU ARE CURRENTLY IN DEBUGGING MODE. REMOVE THIS LINE IF NOT!!')
#    y_normscale = 36.43879218007172 # for 2 million and 5 million
    y_normscale = 36.438613192970415 # for 1 million
        #y_normscale = 36.43879218007172
    if params['load_by_chunks'] == True:
        y_normscale = 36.43879218007172

    # load the noisy testing data back in
    x_data_test, y_data_test_noisefree, y_data_test,_,snrs_test = load_data(params['test_set_dir'],params['inf_pars'],load_condor=True)

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    # reshape arrays for multi-detector
    y_data_train = y_data_train.reshape(y_data_train.shape[0]*y_data_train.shape[1],y_data_train.shape[2]*y_data_train.shape[3])
#    y_data_train = y_data_train.reshape(y_data_train.shape[1],y_data_train.shape[2]*y_data_train.shape[3])
    y_data_test = y_data_test.reshape(y_data_test.shape[0],y_data_test.shape[1]*y_data_test.shape[2])
    y_data_test_noisefree = y_data_test_noisefree.reshape(y_data_test_noisefree.shape[0],y_data_test_noisefree.shape[1]*y_data_test_noisefree.shape[2])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    # load up the posterior samples (if they exist)
    # load generated samples back in
    post_files = []
    #~/bilby_outputs/bilby_output_dynesty1/multi-modal3_0.h5py

    # choose directory with lowest number of total finished posteriors
    num_finished_post = int(1e8)
    for i in params['samplers']:
        if i == 'vitamin':
            continue
        for j in range(1):
            input_dir = '%s_%s%d/' % (params['pe_dir'],i,j+1)
            if type("%s" % input_dir) is str:
                dataLocations = ["%s" % input_dir]

            filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))
            if len(filenames) < num_finished_post:
                sampler_loc = i + str(j+1)

    samp_posteriors = {}
    for samp_idx in params['samplers'][1:]:
        dataLocations_try = '%s_%s' % (params['pe_dir'],sampler_loc)
        dataLocations = '%s_%s' % (params['pe_dir'],samp_idx+'1')
        #for i,filename in enumerate(glob.glob(dataLocations[0])):
        i_idx = 0
        i = 0
        i_idx_use = []
        x_data_test_unnorm = np.copy(x_data_test)
        while i_idx < params['r']*params['r']:

#    for i in range(params['r']*params['r']):
            filename_try = '%s/%s_%d.h5py' % (dataLocations_try,params['bilby_results_label'],i)
            filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)

            # If file does not exist, skip to next file
            try:
                h5py.File(filename_try, 'r')
            except Exception as e:
                i+=1
                print(e)
                continue

            print(filename)
            post_files.append(filename)
            data_temp = {}
            #bounds = {}
            n = 0
            for q in params['inf_pars']:
                 p = q + '_post'
                 par_min = q + '_min'
                 par_max = q + '_max'
                 data_temp[p] = h5py.File(filename, 'r')[p][:]
                 #bounds[par_max] = h5py.File(filename, 'r')[par_max][...].item()
                 #bounds[par_min] = h5py.File(filename, 'r')[par_min][...].item()
                 if p == 'geocent_time_post':
                     data_temp[p] = data_temp[p] - params['ref_geocent_time']
#                 Nsamp = data_temp[p].shape[0]
#                 n = n + 1
#                 continue
                 data_temp[p] = (data_temp[p] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
                 Nsamp = data_temp[p].shape[0]
                 n = n + 1
            XS = np.zeros((Nsamp,n))
            j = 0
            for p,d in data_temp.items():
                XS[:,j] = d
                j += 1

        # Make corner plot of VItamin posterior samples
#        figure = corner.corner(XS, labels=params['inf_pars'],
#                       quantiles=[0.16, 0.5, 0.84],
#                       #range=[[0.0,1.0]]*np.shape(x_data_test)[1],
#                       truths=x_data_test[i,:],
#                       show_titles=True, title_kwargs={"fontsize": 12})

            if i_idx == 0:
                XS_all = np.expand_dims(XS[:params['n_samples'],:], axis=0)
            else:
                # save all posteriors in array
                try:
                    XS_all = np.vstack((XS_all,np.expand_dims(XS[:params['n_samples'],:], axis=0)))
                except ValueError:
                    print(XS_all.shape,np.expand_dims(XS[:params['n_samples'],:], axis=0).shape)
                    exit()

            for q_idx,q in enumerate(params['inf_pars']):
                par_min = q + '_min'
                par_max = q + '_max'

            # rescale parameters back to their physical values
#            if par_min == 'geocent_time_min':
#                continue

                x_data_test_unnorm[i_idx,q_idx] = (x_data_test_unnorm[i_idx,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]

#        plt.savefig('%s/latest_%s/truepost_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))
            i_idx_use.append(i_idx)
            i+=1
            i_idx+=1

        samp_posteriors[samp_idx+'1'] = XS_all

    x_data_test = x_data_test[i_idx_use,:]
    x_data_test_unnorm = x_data_test_unnorm[i_idx_use,:]
    y_data_test = y_data_test[i_idx_use,:]
    y_data_test_noisefree = y_data_test_noisefree[i_idx_use,:]

    # reshape y data into channels last format for convolutional approach
    y_data_test_copy = np.zeros((y_data_test.shape[0],params['ndata'],len(fixed_vals['det'])))
    if params['reduce'] == True or params['n_conv'] != None:
        for i in range(y_data_test.shape[0]):
            for j in range(len(fixed_vals['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_test_copy[i,:,j] = y_data_test[i,idx_range]
        y_data_test = y_data_test_copy
#        y_data_test = y_data_test.reshape(y_data_test.shape[0],params['ndata'],len(fixed_vals['det']))

    VI_pred_all = []
    make_corner_plots = False

    if params['by_channel'] == False:
        y_data_test_new = []
        for sig in y_data_test:
            y_data_test_new.append(sig.T)
        y_data_test = np.array(y_data_test_new)
        del y_data_test_new

    for i in range(params['r']*params['r']):

        if make_corner_plots == False:
            break

        # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
        if params['reduce'] == True or params['n_conv'] != None:
             VI_pred, _, _, dt,_  = VICI_inverse_model.run(params, np.expand_dims(y_data_test[i],axis=0), np.shape(x_data_test)[1],
                                                         y_normscale,
                                                         "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
        else:
            VI_pred, _, _, dt,_  = VICI_inverse_model.run(params, y_data_test[i].reshape([1,-1]), np.shape(x_data_test)[1],
                                                         y_normscale,
                                                         "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
    
        # Make single waveform w/multiple noise real mode weight histogram
        """    
        if i == 0:            
            mode_weights_all = []
            # Iterate over specified number of noise realizations
            for n in range(100):
                # Make new noise realization of test waveform
                y_data_mode_test = y_data_test_noisefree[i] + np.random.normal(0,1,size=(1,int(params['ndata']*len(fixed_vals['det']))))

                # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                _, _, _, _, mode_weights  = VICI_inverse_model.run(params, y_data_mode_test.reshape([1,-1]), np.shape(x_data_test)[1],
                                                                        y_normscale,
                                                                        "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
                mode_weights_all.append([mode_weights])

            mode_weights_all = np.array(mode_weights_all)
            mode_weights_all = mode_weights_all.reshape((mode_weights_all.shape[0]*mode_weights_all.shape[2],mode_weights_all.shape[3]))

            # plot the weights mult noise histogram
            try:
                density_flag = False
                plt.figure()
                for c in range(params['n_modes']):
                    plt.hist(mode_weights_all[:,c],25,alpha=0.5,density=density_flag)
                plt.xlabel('iteration')
                plt.ylabel('KL')
                #plt.legend()
                plt.savefig('%s/latest_%s/mixweights_%s_MultNoiseSingleWave_linear.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.close()
            except:
                pass

            # plot the weights mult noise histogram

            try:
                plt.figure()
                for c in range(params['n_modes']):
                    plt.hist(mode_weights_all[:,c],25,density=density_flag,alpha=0.5,label='component %d' % c)
                plt.xlabel('Mixture weight')
                plt.ylabel('p(w)')
                plt.legend()
                plt.savefig('%s/latest_%s/mixweights_%s_MultNoiseSingleWave_log.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.close()
            except:
                pass
            print('Made multiple noise real mode plots')
        """
        # Make hunter corner plots
#        plotter = plots.make_plots(params,XS_all,np.expand_dims(VI_pred,axis=0),np.expand_dims(x_data_test_unnorm[i],axis=0))        
#        plotter.make_corner_plot(y_data_test_noisefree[i,:params['ndata']],y_data_test[i,:params['ndata']],bounds,i,0,sampler='dynesty1')

        # Make corner corner plots
        bins=50
        #area = sum(numpy.diff(bins)*values)
        defaults_kwargs = dict(
                    bins=bins, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=[0.16, 0.84],
                    levels=(0.68,0.90,0.95), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

        matplotlib.rc('text', usetex=True)                
        parnames = []
        samp_posteriors
        for k_idx,k in enumerate(params['rand_pars']):
            if np.isin(k, params['inf_pars']):
                parnames.append(params['cornercorner_parnames'][k_idx])
        figure = corner.corner(samp_posteriors['cpnest1'][i],**defaults_kwargs,labels=parnames,
                       color='tab:blue',
                       truths=x_data_test[i,:],
                       show_titles=True)
        figure = corner.corner(samp_posteriors['dynesty1'][i], **defaults_kwargs, labels=parnames,
                           color='tab:green',
                           show_titles=True, fig=figure)#, weights=weights)
#        figure = corner.corner(samp_posteriors['emcee1'][i], **defaults_kwargs, labels=parnames,
#                           color='tab:orchid',
#                           show_titles=True, fig=figure)#, weights=weights)
#        figure = corner.corner(samp_posteriors['ptemcee1'][i], **defaults_kwargs, labels=parnames,
#                           color='tab:turquoise',
#                           show_titles=True, fig=figure)#, weights=weights)

        corner.corner(VI_pred, **defaults_kwargs, labels=parnames,
                           color='tab:red', fill_contours=True,
                           show_titles=True, fig=figure)#, weights=weights)
        # compute weights, otherwise the 1d histograms will be different scales, could remove this
        #weights = np.ones(len(VI_pred)) * (len(XS_all[i]) / len(VI_pred))

        left, bottom, width, height = [0.6, 0.69, 0.3, 0.19]
        ax2 = figure.add_axes([left, bottom, width, height])

        # plot waveform in upper-right hand corner
        ax2.plot(np.linspace(0,1,params['ndata']),y_data_test_noisefree[i,:params['ndata']],color='cyan',zorder=50)
#        snr = 'No SNR info'
        snr = round(snrs_test[i,0],2)
        if params['reduce'] == True or params['n_conv'] != None:
            if params['by_channel'] == False:
                 ax2.plot(np.linspace(0,1,params['ndata']),y_data_test[i,0,:params['ndata']],color='darkblue',label='SNR: '+str(snr))
            else:
                ax2.plot(np.linspace(0,1,params['ndata']),y_data_test[i,:params['ndata'],0],color='darkblue',label='SNR: '+str(snr))
        else:
            ax2.plot(np.linspace(0,1,params['ndata']),y_data_test[i,:params['ndata']],color='darkblue',label='SNR: '+str(snr))
        ax2.set_xlabel(r"$\textrm{time (seconds)}$",fontsize=11)
        ax2.yaxis.set_visible(False)
        ax2.tick_params(axis="x", labelsize=11)
        ax2.tick_params(axis="y", labelsize=11)
        ax2.set_ylim([-6,6])
        ax2.grid(False)
        ax2.margins(x=0,y=0)
        ax2.legend()

        plt.savefig('%s/latest_%s/corner_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))
        plt.close()
        del figure
        print('Made corner plot: %s' % str(i+1))
#        del plotter
        
        VI_pred_all.append(VI_pred)

    VI_pred_all = np.array(VI_pred_all)

    # Generate final results plots
    plotter = plots.make_plots(params,XS_all,VI_pred_all,x_data_test)
    

    # Make KL plot
    plotter.gen_kl_plots(VICI_inverse_model,y_data_test,x_data_train,y_normscale,bounds,snrs_test)
#    exit()

    # Make pp plot
    plotter.plot_pp(VICI_inverse_model,y_data_test,x_data_train,0,y_normscale,x_data_test,bounds)
#    exit()


