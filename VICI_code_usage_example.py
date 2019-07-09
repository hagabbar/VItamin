#######################################################################################################################

# -- Example Code for using the Variational Inference for Computational Imaging (VICI) Model --

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

from Models import VICI_forward_model
from Models import VICI_inverse_model
from Models import CVAE
from Neural_Networks import batch_manager
#from Observation_Models import simulate_observations
from data import chris_data
import plots
import bilby_pe

run_label='gpu0',            # label for run
plot_dir="/home/hunter.gabbard/public_html/CBC/VItamin/gw_results/%s" % run_label,                 # plot directory
ndata=256                    # y dimension size
load_train_set = True      # if True, load previously made train samples.
load_test_set = True       # if True, load previously made test samples (including bilby posterior)
T = 1           # length of time series (s)
dt = T/ndata        # sampling time (Sec)
fnyq = 0.5/dt   # Nyquist frequency (Hz)
tot_dataset_size=int(5e5)
tset_split=10000
r = 5                      # the grid dimension for the output tests
iterations=int(1e7)
n_noise=1

# Defining the list of parameter that need to be fed into the models
def get_params():
    params = dict(
        image_size = [1,ndata], # Images Size
        print_values=True, # optionally print values every report interval
        n_samples = 5000, # number of posterior samples to save per reconstruction upon inference 
        num_iterations=20001, # number of iterations inference model (inverse reconstruction)
        initial_training_rate=0.0001, # initial training rate for ADAM optimiser inference model (inverse reconstruction)
        batch_size=1000, # batch size inference model (inverse reconstruction)
        report_interval=500, # interval at which to save objective function values and optionally print info during inference training
        z_dimension=100, # number of latent space dimensions inference model (inverse reconstruction)
        n_weights = 1024, # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        save_interval=1000, # interval at which to save inference model weights
        num_iterations_fw= 3000001, # number of iterations of multifidelity forward model training
        initial_training_rate_fw=0.00002, # initial training rate for ADAM optimiser of multifidelity forward model training
        report_interval_fw=5000, # interval at which to save objective function values and optionally print info during multifidelity forward model training
        z_dimensions_fw = 100, # latent space dimensionality of forward model
        n_weights_fw = 3000, # intermediate layers dimensionality in forward model neural networks
        batch_size_fw=100, # batch size of multifidelity forward model training
        save_interval_fw=5000, # interval at which to save multi-fidelity forward model weights

        
        ndata = ndata,
        r = r,                      # the grid dimension for the output tests
        ndim_x=2,                    # number of parameters to PE on
        sigma=1.0,                   # stadnard deviation of the noise on signal
        usepars=[0,1],             # which parameters you want to do PE on
        tot_dataset_size=tot_dataset_size, # total size of training set
        tset_split=tset_split,            # n_samples per training set file
        seed=42,                     # random seed number
        run_label=run_label,            # label for run
        plot_dir=plot_dir,                 # plot directory
        parnames=['mc','t0'],    # parameter names
        T = T,           # length of time series (s)
        dt = T/ndata,        # sampling time (Sec)
        fnyq = 0.5/dt,   # Nyquist frequency (Hz),
        train_set_dir='training_sets_nowin/tset_tot-%d_split-%d_%dNoise' % (tot_dataset_size,tset_split,n_noise), #location of training set
        test_set_dir='test_sets_nowin/tset_tot-%d_freq-%d' % (r*r,ndata), #location of training set
        add_noise_real=True, # whether or not to add extra noise realizations in training set
        n_noise=n_noise,          # number of noise realizations
        ref_gps_time=1126259643.0, # reference gps time + 0.5s (t0 where test samples are injected+0.5s)
        do_normscale=True   # if true normalize parameters
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
f = open("params_%s.txt" % params['run_label'],"w")
f.write( str(params) )
f.close()

# Make training samples
if not load_train_set:
    # make training set directory
    os.system('mkdir -p %s' % params['train_set_dir'])

    x_data_train_h, y_data_train_lh = [], []
    for i in range(0,params['tot_dataset_size'],params['tset_split']):
        signal_train_images, sig, signal_train_pars = bilby_pe.run(sampling_frequency=params['ndata'],N_gen=params['tset_split'],make_train_samp=True,make_test_samp=False,make_noise=params['add_noise_real'],n_noise=params['n_noise'])

        # scale t0 par to be between 0 and 1
        signal_train_pars[:,2] = params['ref_gps_time'] - signal_train_pars[:,2]

        x_data_train_h.append(signal_train_images)
        y_data_train_lh.append(signal_train_pars)

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
        bilby_pe.run(run_label='test_samp_%d' % i,make_test_samp=True,make_train_samp=False,duration=params['T'],sampling_frequency=params['ndata'],outdir=params['test_set_dir'])

# load test samples
pos_test, labels_test, sig_test = [], [], []
samples = np.zeros((params['r']*params['r'],params['n_samples'],3))
cnt=0
for i in range(params['r']):
    for j in range(params['r']):
        # TODO: remove this bandaged phase file calc
        f = h5py.File('%s/test_samp_%d.h5py' % (params['test_set_dir'],cnt), 'r+')

        # select samples from posterior randomly
        shuffling = np.random.permutation(f['mc_post'][:].shape[0])
        phase = f['phase_post'][:][shuffling]
        mc = f['mc_post'][:][shuffling]
        t0 = params['ref_gps_time'] - f['geocent_time_post'][:][shuffling]
        #dist=f['luminosity_distance_post'][:][shuffling]
        f_new=np.array([mc,phase,t0]).T
        f_new=f_new[:params['n_samples'],:]
        samples[cnt,:,:]=f_new

        # get true scalar parameters
        mc = np.array(f['mc'])
        #if params['do_normscale']:
        #    pos_test.append([mc/normscales[0],
        #                     np.array(f['phase'])/normscales[1],params['ref_gps_time']-np.array(f['geocent_time'])/normscales[2]])
        pos_test.append([mc,np.array(f['phase']),params['ref_gps_time']-np.array(f['geocent_time'])])
        labels_test.append([np.array(f['noisy_waveform'])])
        sig_test.append([np.array(f['noisefree_waveform'])])
        cnt += 1
        f.close()

pos_test = np.array(pos_test)
y_data_test_h = np.array(labels_test).reshape(int(r*r),ndata)
sig_test = np.array(sig_test).reshape(int(r*r),ndata) # noise-free y_test

# Make test samples and posteriors
#for i in range(params['r'] * params['r']):
#    f = h5py.File('%s/test_samp_%d.h5py' % (params['test_set_dir'],i), 'r+')
#    bilby_pe.run(cnt=i,pos_test=pos_test,file_test=f,run_label='test_samp_%d' % i,make_test_samp=True,make_train_samp=False,duration=params['T'],sampling_frequency=params['ndata'],outdir=params['test_set_dir'])
#    f.close()
#print(pos_test)
#exit()

# load generated samples back in
if type("%s" % params['train_set_dir']) is str:
    dataLocations = ["%s" % params['train_set_dir']]
    data={'x_data_train_h': [], 'y_data_train_lh': [], 'y_data_test_h': []}
for filename in os.listdir(dataLocations[0]):
    data_temp={'x_data_train_h': h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data_train_h'][:],
              'y_data_train_lh': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_train_lh'][:]}
    data['x_data_train_h'].append(data_temp['x_data_train_h'])
    data['y_data_train_lh'].append(data_temp['y_data_train_lh'])

data['x_data_train_h'] = np.concatenate(np.array(data['x_data_train_h']), axis=0)
data['y_data_train_lh'] = np.concatenate(np.array(data['y_data_train_lh']), axis=0)

if params['do_normscale']:
    normscales = [np.max(data['x_data_train_h'][:,0]),np.max(data['x_data_train_h'][:,1]),np.max(data['x_data_train_h'][:,2])]#,np.max(data['pos'][:,3])]
    data['x_data_train_h'][:,0]=data['x_data_train_h'][:,0]/normscales[0]
    data['x_data_train_h'][:,1]=data['x_data_train_h'][:,1]/normscales[1]
    data['x_data_train_h'][:,2]=data['x_data_train_h'][:,2]/normscales[2]

x_data_train_h = data['x_data_train_h']
y_data_train_lh = data['y_data_train_lh']
# Remove phase parameter
pos_test = pos_test[:,[0,2]]
x_data_train_h = x_data_train_h[:,[0,2]]
samples = samples[:,:,[0,2]]
if params['do_normscale']: normscales = [normscales[0],normscales[2]]
x_data_train, y_data_train_l, y_data_train_h = x_data_train_h, y_data_train_lh, y_data_train_lh


# Make directory for plots
#plots.make_dirs(params['plot_dir'][0])
# Declare plot class variables
plotter = plots.make_plots(params,samples,None,pos_test)

# First, we learn a multi-fidelity model that lerns to infer high-fidelity (accurate) observations from trget images/objects and low fidelity simulated observations. for this we use the portion of the training set for which we do have real/high fidelity observations.
#forward_loss1, forward_loss2 = VICI_forward_model.train(params, x_data_train_h, y_data_train_h, y_data_train_lh, "forward_model_dir_%s/forward_model.ckpt" % params['run_label'], plotter, pos_test, y_data_test_h, sig_test) # This trains the forward model and saves the weights in forward_model_dir/forward_model.ckpt

#loss_inv=[0.0]
#kl_inv=[0.0]

# We then train the inference model using all training images and associated low-fidelity (inaccurate) observations. Using the previously trained forward model to draw from the observation likelihood.
#loss_inv, kl_inv = VICI_inverse_model.train(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "forward_model_dir_%s/forward_model.ckpt" % params['run_label'], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'], plotter, y_data_test_h) # This trains the inverse model to recover posteriors using the forward model weights stored in forward_model_dir/forward_model.ckpt and saves the inverse model weights in inverse_model_dir/inverse_model.ckpt 
#inverse_loss1, inverse_loss2 = VICI_inverse_model.resume_training(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "forward_model_dir_%s/forward_model.ckpt" % params['run_label'], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])

# The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
#xm, xsx, XS, pmax = VICI_inverse_model.run(params, y_data_test_h, np.shape(x_data_train)[1], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label']) # This runs the trained model using the weights stored in inverse_model_dir/inverse_model.ckpt
# The outputs are the following:
# - xm = marginal means
# - xsx = marginal standard deviations
# - XS = draws from the posterior (3D array with different samples for the same input along the third dimension)
# - pmax = approximate maxima (approximate 'best' reconstructions)

olvec = np.zeros((params['r'],params['r'],int(iterations/(params['num_iterations']-1))))
s=0

for i in range(int(iterations/(params['num_iterations']-1))):
    print('iteration %d' % (i*int(params['num_iterations']-1)))
    if i == 0:
        loss_inv, kl_inv = VICI_inverse_model.train(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "forward_model_dir_%s/forward_model.ckpt" % params['run_label'], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'], plotter, y_data_test_h) # This trains the inverse model to recover posteriors using the forward model weights stored in forward_model_dir/forward_model.ckpt and saves the inverse model weights in inverse_model_dir/inverse_model.ckpt
        #loss_inv, kl_inv = VICI_inverse_model.resume_training(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "forward_model_dir_%s/forward_model.ckpt" % params['run_label'], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])

        olvec = np.zeros((params['r'],params['r'],int(iterations/(params['num_iterations']-1))))
        s=0
    else:
        loss_temp, kl_temp = VICI_inverse_model.resume_training(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "forward_model_dir_%s/forward_model.ckpt" % params['run_label'], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])

        loss_inv = np.concatenate((loss_inv,loss_temp),axis=0)
        kl_inv = np.concatenate((kl_inv,kl_temp),axis=0)

    plotter.make_loss_plot(kl_inv,loss_inv,(int(params['num_iterations']-1)*i))

    # Make overlap plots every 10000 epochs
    if (i*int(params['num_iterations']-1)) % (params['num_iterations']-1) == 0:
        # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
        xm, xsx, XS, pmax = VICI_inverse_model.run(params, y_data_test_h, np.shape(x_data_train)[1], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label']) # This runs the trained model using the weights stored in inverse_model_dir/inverse_model.ckpt

        # Convert XS back to unnormalized version
        if params['do_normscale']:
            for m in range(params['ndim_x']):
                XS[:,m,:] = XS[:,m,:]*normscales[m]

        # Generate final results plots
        plotter = plots.make_plots(params,samples,XS,pos_test)

        # Geneerate overlap scatter plots
        s,olvec = plotter.make_overlap_plot((i*int(params['num_iterations']-1)),iterations,s,olvec)

"""
# Convert XS back to unnormalized version
if params['do_normscale']:
    for m in range(params['ndim_x']):
        XS[:,m,:] = XS[:,m,:]*normscales[m]

# wrap phase estimates around. Deals with phase wrapping.
#XS[:,1,:] =np.remainder(XS[:,1,:],np.pi)

# Generate final results plots
plotter = plots.make_plots(params,samples,XS,pos_test)

# Geneerate overlap scatter plots
plotter.make_overlap_plot(0,iterations,s,olvec)
"""
