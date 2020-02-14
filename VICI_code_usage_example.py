######################################################################################################################

# -- Variational Inference for Gravitational wave Parameter Estimation --


#######################################################################################################################

import argparse
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
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
import corner
import glob

from Models import VICI_inverse_model
from Models import CVAE
from bilby_pe import run
from Neural_Networks import batch_manager
from data import chris_data
import plots

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
os.environ["CUDA_VISIBLE_DEVICES"]="2"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Defining the list of parameter that need to be fed into the models
def get_params():
    ndata = 1024 # length of input to NN == fs * num_detectors
    rand_pars = ['mass_1','mass_2','luminosity_distance','geocent_time','phase','theta_jn','psi','ra','dec']
    run_label = 'multi-modal_%ddet_%dpar_%dHz_run2' % (len(fixed_vals['det']),len(rand_pars),ndata)
    bilby_results_label = '%dpar_%dHz_%ddet_case' % (len(rand_pars),ndata,len(fixed_vals['det']))
    r = 1
    tot_dataset_size = int(1e5)    # total number of training samples to use
    tset_split = int(1e3)          # number of training samples per saved data files
    ref_geocent_time=1126259642.5   # reference gps time
    params = dict(
        ndata = ndata,
        image_size = [1,ndata],        # Images Size
        run_label=run_label,            # label for run
        bilby_results_label=bilby_results_label, # label given to results for bilby posteriors
        tot_dataset_size = tot_dataset_size,
        tset_split = tset_split, 
        plot_dir="/home/hunter.gabbard/public_html/CBC/VItamin/gw_results/%s" % run_label,                 # plot directory
        print_values=True,            # optionally print values every report interval
        n_samples = 1000,             # number of posterior samples to save per reconstruction upon inference 
        num_iterations=int(1e8)+1,    # number of iterations inference model (inverse reconstruction)
        initial_training_rate=0.0001, # initial training rate for ADAM optimiser inference model (inverse reconstruction)
        batch_size=512,               # batch size inference model (inverse reconstruction)
        report_interval=500,          # interval at which to save objective function values and optionally print info during inference training
        save_interval=35000,           # interval at which to save inference model weights
        plot_interval=35000,           # interval over which plotting is done
        z_dimension=48,                # 24 number of latent space dimensions inference model (inverse reconstruction)
        n_weights_r1 = 1024,             # 512 number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        n_weights_r2 = 1024,             # 512 number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        n_weights_q = 1024,             # 512 number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        duration = 1.0,               # the timeseries length in seconds
        r = r,                                # the grid dimension for the output tests
        rand_pars=rand_pars,
        corner_parnames = ['m_{1}\,(\mathrm{M}_{\odot})','m_{2}\,(\mathrm{M}_{\odot})','d_{\mathrm{L}}\,(\mathrm{Mpc})','t_{0}\,(\mathrm{seconds})','{\phi}','\Theta_{jn}','{\psi}','\mathrm{RA}','\mathrm{DEC}'],
        ref_geocent_time=ref_geocent_time,            # reference gps time
        training_data_seed=43,                              # random seed number
        testing_data_seed=44,
        inf_pars=['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','ra','dec'],#,'geocent_time','phase','theta_jn','psi'], # parameter names
        train_set_dir='/home/hunter.gabbard/CBC/VItamin/training_sets_second_sub_%ddet_%dpar_%dHz/tset_tot-%d_split-%d' % (len(fixed_vals['det']),len(rand_pars),ndata,tot_dataset_size,tset_split), #location of training set
        test_set_dir='/home/hunter.gabbard/CBC/VItamin/condor_runs_second_paper_sub/%dpar_%dHz_%ddet_case/test_waveforms' % (len(rand_pars),ndata,len(fixed_vals['det'])), #location of test set
        pe_dir='/home/hunter.gabbard/CBC/VItamin/condor_runs_second_paper_sub/%dpar_%dHz_%ddet_case/test' % (len(rand_pars),ndata,len(fixed_vals['det'])),    # location of bilby PE results
        KL_cycles = 1,                                                         # number of cycles to repeat for the KL approximation
        load_plot_data=False,                                                  # use old plotting data
        samplers=['vitamin', 'dynesty'],#,'emcee','ptemcee','cpnest'],          # samplers to use when plotting

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

def load_data(input_dir,inf_pars,load_condor=False):
 
    # load generated samples back in
    train_files = []
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': []}

    if load_condor == True:
        filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    else:
        filenames = os.listdir(dataLocations[0])
    for filename in filenames:
        train_files.append(filename)
        data_temp={'x_data': h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data'][:],
              'y_data_noisefree': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisefree'][:],
              'y_data_noisy': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisy'][:],
              'rand_pars': h5py.File(dataLocations[0]+'/'+filename, 'r')['rand_pars'][:]}
        data['x_data'].append(data_temp['x_data'])
        data['y_data_noisefree'].append(np.expand_dims(data_temp['y_data_noisefree'], axis=0))
        data['y_data_noisy'].append(np.expand_dims(data_temp['y_data_noisy'], axis=0))
        data['rand_pars'] = data_temp['rand_pars']


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

    return x_data, y_data, y_data_noisy, y_normscale

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

# Get training/test data and parameters of run
params=get_params()
f = open("params_%s.txt" % params['run_label'],"w")
f.write( str(params) )
f.close()

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

        # plot snr distribution
        for j in range(len(fixed_vals['det'])):
            plt.hist(snrs[:,j],bins=20, label=fixed_vals['det'][j], alpha=0.65)
        plt.xlabel('Optimal SNR')
        plt.legend()
        plt.savefig('%s/latest_%s/training_snr_dist.png' % (params['plot_dir'],params['run_label']))
        plt.close()
        print('Generated SNR distribution on training set ...')

#    exit(0)

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

#    exit(0)

# if we are now training the network
if args.train:

    # load the noisefree training data back in
    x_data_train, y_data_train, _, y_normscale = load_data(params['train_set_dir'],params['inf_pars'])

    # load the noisy testing data back in
    x_data_test, y_data_test_noisefree, y_data_test,_ = load_data(params['test_set_dir'],params['inf_pars'],load_condor=True)

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
    dataLocations = '%s_%s1' % (params['pe_dir'],params['samplers'][1])
    #for i,filename in enumerate(glob.glob(dataLocations[0])):
    i_idx = 0
    i = 0
    i_idx_use = []
    while i_idx < params['r']*params['r']:
#    for i in range(params['r']*params['r']):
        filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)

        # If file does not exist, skip to next file
        try:
            h5py.File(filename, 'r')
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
        figure = corner.corner(XS, labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84],
                       #range=[[0.0,1.0]]*np.shape(x_data_test)[1],
                       truths=x_data_test[i,:],
                       show_titles=True, title_kwargs={"fontsize": 12})

        if i_idx == 0:
            XS_all = np.expand_dims(XS[:params['n_samples'],:], axis=0)
        else:
            # save all posteriors in array
            XS_all = np.vstack((XS_all,np.expand_dims(XS[:params['n_samples'],:], axis=0)))


        for q_idx,q in enumerate(params['inf_pars']):
            par_min = q + '_min'
            par_max = q + '_max'

            # rescale parameters back to their physical values
#            if par_min == 'geocent_time_min':
#                continue

            x_data_test[i,q_idx] = (x_data_test[i,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]

        plt.savefig('%s/latest_%s/truepost_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))
        i_idx_use.append(i)
        i+=1
        i_idx+=1


    y_data_test = y_data_test[i_idx_use,:]
    y_data_test_noisefree = y_data_test_noisefree[i_idx_use,:]
    x_data_test = x_data_test[i_idx_use,:]

    VICI_inverse_model.train(params, x_data_train, y_data_train,
                             x_data_test, y_data_test, y_data_test_noisefree,
                             y_normscale,
                             "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'],
                             x_data_test, bounds, fixed_vals,
                             XS_all) 

# if we are now testing the network
if args.test:

    # load the noisefree training data back in
    x_data_train, y_data_train, _, y_normscale = load_data(params['train_set_dir'],params['inf_pars'])

    # load the noisy testing data back in
    x_data_test, y_data_test_noisefree, y_data_test,_ = load_data(params['test_set_dir'],params['inf_pars'],load_condor=True)

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
    dataLocations = '%s_%s1' % (params['pe_dir'],params['samplers'][1])
    #for i,filename in enumerate(glob.glob(dataLocations[0])):
    i_idx = 0
    i = 0
    i_idx_use = []
    while i_idx < params['r']*params['r']:

    #for i in range(params['r']*params['r']):
        filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)

        # If file does not exist, skip to next file
        try:
            h5py.File(filename, 'r')
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
        figure = corner.corner(XS, labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84],
                       #range=[[0.0,1.0]]*np.shape(x_data_test)[1],
                       truths=x_data_test[i,:],
                       show_titles=True, title_kwargs={"fontsize": 12})

        if i_idx == 0:
            XS_all = np.expand_dims(XS[:params['n_samples'],:], axis=0)
        else:
            # save all posteriors in array
            XS_all = np.vstack((XS_all,np.expand_dims(XS[:params['n_samples'],:], axis=0)))


        for q_idx,q in enumerate(params['inf_pars']):
            par_min = q + '_min'
            par_max = q + '_max'

            # rescale parameters back to their physical values
#            if par_min == 'geocent_time_min':
#                continue

            x_data_test[i,q_idx] = (x_data_test[i,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]

        plt.savefig('%s/latest_%s/truepost_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))
        i_idx_use.append(i)
        i+=1
        i_idx+=1

    VI_pred_all = []
    for i in range(params['r']*params['r']):
        # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
        VI_pred, dt  = VICI_inverse_model.run(params, np.expand_dims(y_data_test[i],axis=0), np.shape(x_data_test)[1],
                                                         y_normscale,
                                                         "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
        VI_pred_all.append(VI_pred)

    VI_pred_all = np.array(VI_pred_all)

    # Generate final results plots
    plotter = plots.make_plots(params,XS_all,VI_pred_all,x_data_test)

    # Make KL plot
#    plotter.gen_kl_plots(VICI_inverse_model,y_data_test,x_data_train,y_normscale)
#    exit()

    # Make pp plot
    plotter.plot_pp(VICI_inverse_model,y_data_test,x_data_train,0,y_normscale,x_data_test,bounds)
    exit()


