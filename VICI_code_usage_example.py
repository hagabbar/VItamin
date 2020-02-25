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

os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--gen_train", default=False, help="generate the training data")
parser.add_argument("--gen_test", default=False, help="generate the testing data")
parser.add_argument("--train", default=False, help="train the network")
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
        'det':'H1'}

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

# Defining the list of parameter that need to be fed into the models
def get_params():
    ndata = 256
    run_label = 'multi-modal90'
    r = 2
    tot_dataset_size = int(1e5)    # total number of training samples to use
    tset_split = int(1e3)          # number of training samples per saved data files
    ref_geocent_time=1126259642.5   # reference gps time
    params = dict(
        ndata = ndata,
        image_size = [1,ndata],        # Images Size
        run_label=run_label,            # label for run
        tot_dataset_size = tot_dataset_size,
        tset_split = tset_split, 
        plot_dir="/data/public_html/chrism/VItamin/gw_results/%s" % run_label,                 # plot directory
        print_values=True,            # optionally print values every report interval
        n_samples = 10000,             # number of posterior samples to save per reconstruction upon inference 
        num_iterations=int(1e8)+1,    # number of iterations inference model (inverse reconstruction)
        initial_training_rate=0.0001, # initial training rate for ADAM optimiser inference model (inverse reconstruction)
        batch_size=512,               # batch size inference model (inverse reconstruction)
        report_interval=500,          # interval at which to save objective function values and optionally print info during inference training
        save_interval=10000,           # interval at which to save inference model weights
        plot_interval=20000,           # interval over which plotting is done
        z_dimension=4,                # number of latent space dimensions inference model (inverse reconstruction)
        n_modes=2,                  # number of modes in the latent space
        n_hlayers=2,                # the number of hidden layers in each network
        n_weights_r1 = 1024,             # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        n_weights_r2 = 1024,             # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        n_weights_q = 1024,             # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        n_conv = 1,
        n_filters = 16,
        filter_size = 4,
        drate = 0.2,
        maxpool = 2,
        duration = 1.0,               # the timeseries length in seconds
        r = r,                                # the grid dimension for the output tests
        rand_pars=['mass_1','mass_2','luminosity_distance','geocent_time','phase'],
        ref_geocent_time=ref_geocent_time,            # reference gps time
        training_data_seed=43,                              # random seed number
        testing_data_seed=44,
        inf_pars=['geocent_time','phase','luminosity_distance'], # parameter names
        wrap_pars=['phase'],                  # parameters that get wrapped on the 1D parameter 
        train_set_dir='/home/chrism/training_sets/tset_tot-%d_split-%d_samp-%d' % (tot_dataset_size,tset_split,ndata), #location of training set
        test_set_dir='/home/chrism/testing_sets/tset_tot-%d_samp-%d' % (r*r,ndata), #location of test set
        pe_dir='/home/chrism/bilby_outputs/bilby_output', #location of bilby PE results
        KL_cycles = 1,                       # number of cycles to repeat for the KL approximation
        #add_noise_real=True,                  # whether or not to add extra noise realizations in training set
        #do_normscale=True,                    # if true normalize parameters
        #do_mc_eta_conversion=False,           # if True, convert m1 and m2 parameters into mc and eta
        #n_kl_samp=int(r*r),                        # number of iterations in statistic tests TODO: remove this
        #do_adkskl_test=False,                  # if True, do statistic tests
        #do_m1_m2_cut=False,                   # if True, make a cut on all m1 and m2 values    
        #do_extra_noise=True,                  # add extra noise realizations during training
        #Npp = int(r*r),                             # number of test signals per pp-plot. TODO: use same 
                                              # use same samples as bilby
        #samplers=['vitamin','dynesty','emcee','ptemcee','cpnest'],          # list of available bilby samplers to use
        #use_samplers = [0,1],                  # number of Bilby samplers to use 
        #kl_set_dir='/home/chrism/kl_output', # location of test set used for kl
        #do_only_test = False,                  # if true, don't train but only run on test samples using pretrained network
        #load_plot_data = False,                # if true, load in previously generated plot data, otherwise generate plots from scratch
        doPE = True,                          # if True then do bilby PE
        #whitening_factor = np.sqrt(float(ndata)) # whitening scale factor
    )
    return params

def load_data(input_dir,inf_pars):
 
    # load generated samples back in
    train_files = []
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': []}
    for filename in os.listdir(dataLocations[0]):
        train_files.append(filename)
        data_temp={'x_data': h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data'][:],
              'y_data_noisefree': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisefree'][:],
              'y_data_noisy': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisy'][:],
              'rand_pars': h5py.File(dataLocations[0]+'/'+filename, 'r')['rand_pars'][:]}
        data['x_data'].append(data_temp['x_data'])
        data['y_data_noisefree'].append(data_temp['y_data_noisefree'])
        data['y_data_noisy'].append(data_temp['y_data_noisy'])
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

    # Iterate over number of requested training samples
    for i in range(0,params['tot_dataset_size'],params['tset_split']):

        _, signal_train, signal_train_pars = run(sampling_frequency=params['ndata']/params['duration'],
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
        hf.close()

# Make testing set directory
if args.gen_test:

    # Make training set directory
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

    # Load test samples
    #pos_test, labels_test, sig_test = [], [], []
    #samples = np.zeros((params['r']*params['r'],params['n_samples'],params['ndim_x']))
    #cnt=0
    #for i in range(params['r']):
    #    for j in range(params['r']):
    #    
    #        # Load test sample file
    #        f = h5py.File('%s/%s_%s.h5py' % (params['test_set_dir'],params['run_label'],str(cnt)), 'r+')
    
    #        # rescale all parameters to 0->1
    #        
    #
    #        # select samples from posterior randomly
    #        phase = (f['phase_post'][:] - (params['prior_min'][1])) / (params['prior_max'][1] - params['prior_min'][1])
    #     
    #        if params['do_mc_eta_conversion']:
    #            m1 = f['mass_1_post'][:]
    #            m2 = f['mass_2_post'][:]
    #            eta = (m1*m2)/(m1+m2)**2
    #            mc = np.sum([m1,m2], axis=0)*eta**(3.0/5.0)
    #        else: 
    #            m1 = (f['mass_1_post'][:] - (params['prior_min'][0])) / (params['prior_max'][0] - params['prior_min'][0])
    #            m2 = (f['mass_2_post'][:] - (params['prior_min'][3])) / (params['prior_max'][3] - params['prior_min'][3])
    #        t0 = (f['geocent_time_post'][:] - (params['prior_min'][2])) / (params['prior_max'][2] - params['prior_min'][2])
    #        dist=(f['luminosity_distance_post'][:] - (params['prior_min'][4])) / (params['prior_max'][4] - params['prior_min'][4])
    #
    #        if params['do_mc_eta_conversion']:
    #            f_new=np.array([mc,phase,t0,eta]).T
    #        else:
    #            f_new=np.array([m1,phase,t0,m2,dist]).T
    #        f_new=f_new[:params['n_samples'],:]
    #        print(f_new.shape)
    #        samples[cnt,:,:]=f_new
    #
    #        # get true scalar parameters
    #        if params['do_mc_eta_conversion']:
    #            m1 = np.array(f['mass_1'])
    #            m2 = np.array(f['mass_2'])
    #            eta = (m1*m2)/(m1+m2)**2
    #            mc = np.sum([m1,m2])*eta**(3.0/5.0)
    #            pos_test.append([mc,np.array(f['phase']),(np.array(f['geocent_time']) - (params['prior_min'][2])) / (params['prior_max'][2] - params['prior_min'][2]),eta])
    #        else:
    #            m1 = (np.array(f['mass_1']) - (params['prior_min'][0])) / (params['prior_max'][0] - params['prior_min'][0])
    #            m2 = (np.array(f['mass_2']) - (params['prior_min'][3])) / (params['prior_max'][3] - params['prior_min'][3])
    #            t0 = (np.array(f['geocent_time']) - (params['prior_min'][2])) / (params['prior_max'][2] - params['prior_min'][2])
    #            dist = (np.array(f['luminosity_distance']) - (params['prior_min'][4])) / (params['prior_max'][4] - params['prior_min'][4])
    #            phase = (np.array(f['phase']) - (params['prior_min'][1])) / (params['prior_max'][1] - params['prior_min'][1])
    #            pos_test.append([m1,phase,t0,m2,dist])
    #        labels_test.append([np.array(f['noisy_waveform'])])
    #        sig_test.append([np.array(f['noisefree_waveform'])])
    #
    #        cnt += 1
    #        f.close()

    # Set test arrays
    #pos_test = np.array(pos_test) # test parameters
    # TODO: move whitening terms to where whitening is done
    #y_data_test_h = np.array(labels_test).reshape(int(r*r),ndata) * params['whitening_factor'] # noisy y test
    #sig_test = np.array(sig_test).reshape(int(r*r),ndata) * params['whitening_factor'] # noise-free y test

    # Get list of training files
    #train_files = []
    #if type("%s" % params['train_set_dir']) is str:
    #    dataLocations = ["%s" % params['train_set_dir']]
    #    data={'x_data_train_h': [], 'y_data_train_lh': [], 'y_data_test_h': []}
    #for filename in os.listdir(dataLocations[0]):
    #    train_files.append(filename)

    # load the noisefree training data back in
    x_data_train, y_data_train, _, y_normscale = load_data(params['train_set_dir'],params['inf_pars'])

    # load the noisy testing data back in
    x_data_test, _, y_data_test,_ = load_data(params['test_set_dir'],params['inf_pars'])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    # load up the posterior samples (if they exist)
    # load generated samples back in
    post_files = []
    #~/bilby_outputs/bilby_output_dynesty1/multi-modal3_0.h5py
    dataLocations = '%s_dynesty1' % params['pe_dir']
    #for i,filename in enumerate(glob.glob(dataLocations[0])):
    for i in range(params['r']*params['r']):
        filename = '%s/%d.h5py' % (dataLocations,i)
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
                       #range=[[-0.1,1.1]]*np.shape(x_data_test)[1],
                       truths=x_data_test[i,:],
                       show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig('%s/latest_%s/truepost_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))

    #if params['load_plot_data'] == False:
    #    y_normscale = [np.max(np.abs(y_data_train_lh))]
    #    hf = h5py.File('plotting_data_%s/y_normscale_value.h5' % params['run_label'], 'w')
    #    hf.create_dataset('y_normscale', data=y_normscale)
    #    hf.close()
    #else:
    #    hf = h5py.File('plotting_data_%s/y_normscale_value.h5' % params['run_label'], 'r')
    #    y_normscale = np.array(hf['y_normscale'])
    #    hf.close()

    #y_data_train_lh /= y_normscale[0]
    #y_data_test_h /= y_normscale[0]
    #sig_test /= y_normscale[0]

    #if params['do_normscale']: 
    #    normscales = [normscales[0],normscales[1],normscales[2],normscales[3],normscales[4]]
    #x_data_train, y_data_train_l, y_data_train_h = x_data_train_h, y_data_train_lh, y_data_train_lh

    # Declare plot class variables
    #plotter = plots.make_plots(params,samples,None,pos_test)

    # Plot test sample time series
    #plotter.plot_testdata((y_data_test),sig_test,params['r']**2,params['plot_dir'])

    # Train model - This trains the inverse model to recover posteriors using the 
    # forward model weights stored in forward_model_dir/forward_model.ckpt and saves 
    # the inverse model weights in inverse_model_dir/inverse_model.ckpt
    VICI_inverse_model.train(params, x_data_train, y_data_train,
                             x_data_test, y_data_test,
                             y_normscale,
                             "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label']) 
    exit(0)

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

    if params['load_plot_data']  == False:
        hf = h5py.File('plotting_data_%s/pos_test.h5' % params['run_label'], 'w')
        hf.create_dataset('pos_test', data=pos_test)
        hf.close()

    # Make KL plot
    plotter.gen_kl_plots(VICI_inverse_model,y_data_test_h,x_data_train,normscales)

    # Make pp plot
    plotter.plot_pp(VICI_inverse_model,y_data_test_h,x_data_train,0,normscales,samples,pos_test)

    # Make corner plots
#    plotter.make_corner_plot(sig_test * y_normscale[0],y_data_test_h * y_normscale[0],sampler='dynesty1')

    # Geneerate overlap scatter plots
#    plotter.make_overlap_plot(0,iterations,s,olvec,olvec_2d,adksVec)

