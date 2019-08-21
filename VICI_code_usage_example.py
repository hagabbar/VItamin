#######################################################################################################################

# -- Variational Inference for Gravitational wave Parameter Estimation --


#######################################################################################################################

import numpy as np
import tensorflow as tf
import scipy.io as sio
import scipy.misc as mis
import h5py
from sys import exit
import os
import bilby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from Models import VICI_inverse_model
from Models import CVAE
from bilby_pe import run
from Neural_Networks import batch_manager
from data import chris_data
import plots

run_label='gpu4',            # label for run
plot_dir="/home/hunter.gabbard/public_html/CBC/VItamin/gw_results/%s" % run_label,                 # plot directory
ndata=256                    # y dimension size
load_train_set = True       # if True, load previously made train samples.
load_test_set = True         # if True, load previously made test samples (including bilby posterior)
T = 1                        # length of time series (s)
dt = T/ndata                 # sampling time (Sec) #TODO: remove this.
fnyq = 0.5/dt                # Nyquist frequency (Hz) #TODO: remove this.
tot_dataset_size=int(1e6)    # total number of training samples to use
tset_split=int(5e4)          # number of training samples per saved data files
r = 5                        # the grid dimension for the output tests
n_noise=1                    # this is a redundant parameter. Needs to be removed TODO
ref_geocent_time=1126259642.5            # reference gps time

# Defining the list of parameter that need to be fed into the models
def get_params():
    params = dict(
        image_size = [1,ndata],       # Images Size
        print_values=True,            # optionally print values every report interval
        n_samples = 5000,             # number of posterior samples to save per reconstruction upon inference 
        num_iterations=int(1e8)+1,        # number of iterations inference model (inverse reconstruction)
        initial_training_rate=0.0001, # initial training rate for ADAM optimiser inference model (inverse reconstruction)
        batch_size=128,               # batch size inference model (inverse reconstruction)
        report_interval=500,          # interval at which to save objective function values and optionally print info during inference training
        z_dimension=32,               # number of latent space dimensions inference model (inverse reconstruction)
        n_weights = 2048,             # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        save_interval=500,           # interval at which to save inference model weights
        plot_interval=100000,           # interval over which plotting is done

        
        ndata = ndata,
        r = r,                                # the grid dimension for the output tests
        ndim_x=4,                             # number of parameters to PE on
        sigma=1.0,                            # stadnard deviation of the noGise on signal
        usepars=[0,1,2,3],                    # which parameters you want to do PE on
        prior_min=[35.0,0.0,ref_geocent_time+0.15,35.0,1000.0],                         # minimum prior range
        prior_max=[80.0,2*np.pi,ref_geocent_time+0.35,80.0,3000.0],                         # maximum prior range
        tot_dataset_size=tot_dataset_size,    # total size of training set
        tset_split=tset_split,                # n_samples per training set file
        seed=42,                              # random seed number
        run_label=run_label,                  # label for run
        plot_dir=plot_dir,                    # plot directory
        parnames=['m1','t0','m2','lum_dist'], # parameter names
        T = T,                                # length of time series (s)
        dt = T/ndata,                         # sampling time (Sec)
        fnyq = 0.5/dt,                        # Nyquist frequency (Hz),
        train_set_dir='training_sets_final/tset_tot-%d_split-%d_%dNoise' % (tot_dataset_size,tset_split,n_noise), #location of training set
        test_set_dir='condor_runs/final_run/bilby_output_dynesty1', #location of test set for all plots ecept kl
        add_noise_real=True,                  # whether or not to add extra noise realizations in training set
        n_noise=n_noise,                      # number of noise realizations
        ref_geocent_time=ref_geocent_time,            # reference gps time 
        do_normscale=True,                    # if true normalize parameters
        do_mc_eta_conversion=False,           # if True, convert m1 and m2 parameters into mc and eta
        n_kl_samp=25,                        # number of iterations in statistic tests TODO: remove this
        do_adkskl_test=True,                  # if True, do statistic tests
        do_m1_m2_cut=False,                   # if True, make a cut on all m1 and m2 values    
        do_extra_noise=True,                  # add extra noise realizations during training
        do_load_in_chunks=False,              # if True, load training samples in random file chucnks every 25000 epochs
        Npp = 25,                             # number of test signals per pp-plot. TODO: use same 
                                              # use same samples as bilby
        samplers=['vitamin','dynesty','emcee'],          # list of available bilby samplers to use
        use_samplers = [0,1,2],                  # number of Bilby samplers to use 
        kl_set_dir='condor_runs/final_run/bilby_output', # location of test set used for kl
        do_only_test = False                 # if true, don't train but only run on test samples using pretrained network
    )
    return params

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
if not load_train_set:
    # Make training set directory
    os.system('mkdir -p %s' % params['train_set_dir'])

    # Iterate over number of requested training samples
#    x_data_train_h, y_data_train_lh = [], []
    for i in range(0,params['tot_dataset_size'],params['tset_split']):
        signal_train_images, sig, signal_train_pars = run(sampling_frequency=params['ndata'],N_gen=params['tset_split'],make_train_samp=True,make_test_samp=False,make_noise=params['add_noise_real'],n_noise=params['n_noise'])

        # Scale t0 par to be between 0 and 1
#        signal_train_pars[:,2] = params['ref_gps_time'] - signal_train_pars[:,2]
        signal_train_pars[:,2] = (signal_train_pars[:,2] - (params['prior_min'][2])) / (params['prior_max'][2] - params['prior_min'][2])

#        x_data_train_h.append(signal_train_images)
#        y_data_train_lh.append(signal_train_pars)

        print("Generated: %s/data_%d-%d_%dNoise.h5py ..." % (params['train_set_dir'],(i+params['tset_split']),params['tot_dataset_size'],params['n_noise']))

        hf = h5py.File('%s/data_%d-%d_%dNoise.h5py' % (params['train_set_dir'],(i+params['tset_split']),params['tot_dataset_size'],params['n_noise']), 'w')
        hf.create_dataset('x_data_train_h', data=signal_train_pars)
        hf.create_dataset('y_data_train_lh', data=signal_train_images)
        hf.create_dataset('y_data_train_noisefree', data=sig)
        hf.create_dataset('parnames', data=np.string_(params['parnames']))
        hf.close()

# Make testing set directory
if not load_test_set:

    # Make test samples and posteriors
    for i in range(params['r'] * params['r']):
        run(run_label='test_samp_%d' % i,make_test_samp=True,make_train_samp=False,duration=params['T'],sampling_frequency=params['ndata'],outdir=params['test_set_dir'])

# Load test samples
pos_test, labels_test, sig_test = [], [], []
samples = np.zeros((params['r']*params['r'],params['n_samples'],params['ndim_x']+1))
cnt=0
for i in range(params['r']):
    for j in range(params['r']):
        # Load test sample file
        f = h5py.File('%s/test_samp_%d.h5py' % (params['test_set_dir'],cnt), 'r+')

        # select samples from posterior randomly
        phase = (f['phase_post'][:] - (params['prior_min'][1])) / (params['prior_max'][1] - params['prior_min'][1])
         
        if params['do_mc_eta_conversion']:
            m1 = f['mass_1_post'][:]
            m2 = f['mass_2_post'][:]
            eta = (m1*m2)/(m1+m2)**2
            mc = np.sum([m1,m2], axis=0)*eta**(3.0/5.0)
        else: 
            m1 = (f['mass_1_post'][:] - (params['prior_min'][0])) / (params['prior_max'][0] - params['prior_min'][0])
            m2 = (f['mass_2_post'][:] - (params['prior_min'][3])) / (params['prior_max'][3] - params['prior_min'][3])
        t0 = (f['geocent_time_post'][:] - (params['prior_min'][2])) / (params['prior_max'][2] - params['prior_min'][2])
        dist=(f['luminosity_distance_post'][:] - (params['prior_min'][4])) / (params['prior_max'][4] - params['prior_min'][4])
#        theta_jn=f['theta_jn_post'][:][shuffling]
        if params['do_mc_eta_conversion']:
            f_new=np.array([mc,phase,t0,eta]).T
        else:
            f_new=np.array([m1,phase,t0,m2,dist]).T
        f_new=f_new[:params['n_samples'],:] # TODO use random integers for indexing.
        samples[cnt,:,:]=f_new

        # get true scalar parameters
        if params['do_mc_eta_conversion']:
            m1 = np.array(f['mass_1'])
            m2 = np.array(f['mass_2'])
            eta = (m1*m2)/(m1+m2)**2
            mc = np.sum([m1,m2])*eta**(3.0/5.0)
            pos_test.append([mc,np.array(f['phase']),(np.array(f['geocent_time']) - (params['prior_min'][2])) / (params['prior_max'][2] - params['prior_min'][2]),eta])
        else:
            m1 = (np.array(f['mass_1']) - (params['prior_min'][0])) / (params['prior_max'][0] - params['prior_min'][0])
            m2 = (np.array(f['mass_2']) - (params['prior_min'][3])) / (params['prior_max'][3] - params['prior_min'][3])
            t0 = (np.array(f['geocent_time']) - (params['prior_min'][2])) / (params['prior_max'][2] - params['prior_min'][2])
            dist = (np.array(f['luminosity_distance']) - (params['prior_min'][4])) / (params['prior_max'][4] - params['prior_min'][4])
            phase = (np.array(f['phase']) - (params['prior_min'][1])) / (params['prior_max'][1] - params['prior_min'][1])
            pos_test.append([m1,phase,t0,m2,dist])
        labels_test.append([np.array(f['noisy_waveform'])])
        sig_test.append([np.array(f['noisefree_waveform'])])

        cnt += 1
        f.close()

# Set test arrays
pos_test = np.array(pos_test) # test parameters
# TODO: move whitening terms to where whitening is done
y_data_test_h = np.array(labels_test).reshape(int(r*r),ndata) * np.sqrt(float(params['ndata'])/2.0) # noisy y test
sig_test = np.array(sig_test).reshape(int(r*r),ndata) * np.sqrt(float(params['ndata'])/2.0) # noise-free y test

# Get list of training files
train_files = []
if type("%s" % params['train_set_dir']) is str:
    dataLocations = ["%s" % params['train_set_dir']]
    data={'x_data_train_h': [], 'y_data_train_lh': [], 'y_data_test_h': []}
for filename in os.listdir(dataLocations[0]):
    train_files.append(filename)

if not params['do_load_in_chunks']:
    # load generated samples back in
    if type("%s" % params['train_set_dir']) is str:
        dataLocations = ["%s" % params['train_set_dir']]
        data={'x_data_train_h': [], 'y_data_train_lh': [], 'y_data_test_h': [], 'y_data_train_noisefree': []}
    for filename in os.listdir(dataLocations[0]):
        data_temp={'x_data_train_h': h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data_train_h'][:],
                  'y_data_train_lh': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_train_lh'][:],
                  'y_data_train_noisefree': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_train_noisefree'][:]}
        data['x_data_train_h'].append(data_temp['x_data_train_h'])
        data['y_data_train_lh'].append(data_temp['y_data_train_lh'])
        data['y_data_train_noisefree'].append(data_temp['y_data_train_noisefree'])

    data['x_data_train_h'] = np.concatenate(np.array(data['x_data_train_h']), axis=0)
    data['y_data_train_lh'] = np.concatenate(np.array(data['y_data_train_lh']), axis=0)
    data['y_data_train_noisefree'] = np.concatenate(np.array(data['y_data_train_noisefree']), axis=0)

    if params['do_normscale']:

        normscales = [1.0,1.0,1.0,1.0,1.0]


        # normalize training set
        data['x_data_train_h'][:,0]=(data['x_data_train_h'][:,0] - (params['prior_min'][0])) / (params['prior_max'][0] - params['prior_min'][0])
        data['x_data_train_h'][:,1]=(data['x_data_train_h'][:,1] - (params['prior_min'][1])) / (params['prior_max'][1] - params['prior_min'][1])
        data['x_data_train_h'][:,2]=data['x_data_train_h'][:,2] #- (params['prior_min'][2])) / (params['prior_max'][2] - params['prior_min'][2])
        data['x_data_train_h'][:,3]=(data['x_data_train_h'][:,3] - (params['prior_min'][3])) / (params['prior_max'][3] - params['prior_min'][3])
        data['x_data_train_h'][:,4]=(data['x_data_train_h'][:,4] - (params['prior_min'][4])) / (params['prior_max'][4] - params['prior_min'][4])
    #    data['x_data_train_h'][:,5]=data['x_data_train_h'][:,5]/normscales[5]

    # TODO: move this to whitening procedure in bilby
    x_data_train_h = data['x_data_train_h']
    y_data_train_lh = data['y_data_train_lh'] * np.sqrt(float(params['ndata'])/2.0)
    y_data_train_noisefree = data['y_data_train_noisefree'] * np.sqrt(float(params['ndata'])/2.0)

    # Remove phase parameter
    pos_test = pos_test[:,[0,2,3,4]]
    x_data_train_h = x_data_train_h[:,[0,2,3,4]]
    samples = samples[:,:,[0,2,3,4]]

    # rescale y data for training/testing by absolute max of training set
#    y_normscale = [13.206409999486425]
    y_normscale = [np.max(np.abs(y_data_train_lh))]
    y_data_train_lh /= y_normscale[0]
    y_data_test_h /= y_normscale[0]

    if params['do_normscale']: 
        normscales = [normscales[0],normscales[2],normscales[3],normscales[4]]#,normscales[5]]
    x_data_train, y_data_train_l, y_data_train_h = x_data_train_h, y_data_train_lh, y_data_train_lh

if params['do_load_in_chunks']:

    if params['do_normscale']:
        normscales = [80.0,2*np.pi,0.6,80.0,3000.0]#,np.max(data['pos'][:,5])]
        normscales = [normscales[0],normscales[2],normscales[3],normscales[4]]#,normscales[5]]

    # Remove phase parameter
    pos_test = pos_test[:,[0,2,3,4]]
    samples = samples[:,:,[0,2,3,4]]

# Make directory for plots
#plots.make_dirs(params['plot_dir'][0])

# Declare plot class variables
plotter = plots.make_plots(params,samples,None,pos_test)

# Plot test sample time series
plotter.plot_testdata((y_data_test_h),sig_test,params['r']**2,params['plot_dir'])

# Train model
if not params['do_only_test']:
    if params['do_load_in_chunks']:
        # Get first set of training samples
        x_data_train, y_data_train_l, y_data_train_h, x_data_train_h, y_data_train_lh =  chris_data.load_training_set(params,train_files,normscales)
        y_data_train_h, y_data_train_lh = y_data_train_h * np.sqrt(2*params['ndata']), y_data_train_lh * np.sqrt(2*params['ndata'])


    loss_inv, kl_inv, train_files = VICI_inverse_model.train(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'], plotter, y_data_test_h,train_files,normscales,y_data_train_noisefree,samples,pos_test,y_normscale) # This trains the inverse model to recover posteriors using the forward model weights stored in forward_model_dir/forward_model.ckpt and saves the inverse model weights in inverse_model_dir/inverse_model.ckpt

# Test model
if params['do_only_test']:
    # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
    xm, xsx, XS, pmax, _ = VICI_inverse_model.run(params, y_data_test_h, np.shape(x_data_train)[1], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label']) # This runs the trained model using the weights stored in inverse_model_dir/inverse_model.ckpt

    # TODO: remove this later
#    normscales = [79.99998770493086, 0.5999996662139893, 79.95626661877512, 2999.9996467099486]

    # Convert XS back to unnormalized version
    if params['do_normscale']:
        for m in range(params['ndim_x']):
            XS[:,m,:] = XS[:,m,:] * normscales[m]

    # Generate final results plots
    plotter = plots.make_plots(params,samples,XS,pos_test)

    # Make corner plots
#    plotter.make_corner_plot(sampler='dynesty1')

    # Make KL plot
#    plotter.gen_kl_plots(VICI_inverse_model,y_data_test_h,x_data_train,normscales)

    # Make pp plot
    plotter.plot_pp(VICI_inverse_model,y_data_test_h,x_data_train,0,normscales,samples,pos_test)

    # Geneerate overlap scatter plots
#    plotter.make_overlap_plot(0,iterations,s,olvec,olvec_2d,adksVec)

