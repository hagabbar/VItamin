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
import time

from Models import VICI_inverse_model
from Models import CVAE
from Neural_Networks import batch_manager
from data import chris_data
import plots

run_label='gpu0',            # label for run
plot_dir="/home/hunter.gabbard/public_html/CBC/VItamin/gw_results/%s" % run_label,                 # plot directory
ndata=256                    # y dimension size
load_train_set = True        # if True, load previously made train samples.
load_test_set = True        # if True, load previously made test samples (including bilby posterior)
T = 1                        # length of time series (s)
dt = T/ndata                 # sampling time (Sec)
fnyq = 0.5/dt                # Nyquist frequency (Hz)
tot_dataset_size=int(1e3)    # total number of training samples to use
tset_split=int(1e3)          # number of training samples per saved data files
r = 5                        # the grid dimension for the output tests
iterations=int(1e7)          # total number of training iterations
n_noise=1                    # this is a redundant parameter. Needs to be removed TODO

# Defining the list of parameter that need to be fed into the models
def get_params():
    params = dict(
        image_size = [1,ndata],       # Images Size
        print_values=True,            # optionally print values every report interval
        n_samples = 5000,             # number of posterior samples to save per reconstruction upon inference 
        num_iterations=150001,        # number of iterations inference model (inverse reconstruction)
        initial_training_rate=0.0001, # initial training rate for ADAM optimiser inference model (inverse reconstruction)
        batch_size=128,               # batch size inference model (inverse reconstruction)
        report_interval=500,          # interval at which to save objective function values and optionally print info during inference training
        z_dimension=64,               # number of latent space dimensions inference model (inverse reconstruction)
        n_weights = 2048,             # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        save_interval=5000,           # interval at which to save inference model weights

        
        ndata = ndata,
        r = r,                                # the grid dimension for the output tests
        ndim_x=4,                             # number of parameters to PE on
        sigma=1.0,                            # stadnard deviation of the noise on signal
        usepars=[0,1,2,3],                    # which parameters you want to do PE on
        tot_dataset_size=tot_dataset_size,    # total size of training set
        tset_split=tset_split,                # n_samples per training set file
        seed=42,                              # random seed number
        run_label=run_label,                  # label for run
        plot_dir=plot_dir,                    # plot directory
        parnames=['m1','t0','m2','lum_dist'], # parameter names
        T = T,                                # length of time series (s)
        dt = T/ndata,                         # sampling time (Sec)
        fnyq = 0.5/dt,                        # Nyquist frequency (Hz),
        train_set_dir='training_sets_nowin_par4_35-80m_1000-3000d/tset_tot-%d_split-%d_%dNoise' % (tot_dataset_size,tset_split,n_noise), #location of training set
        test_set_dir='test_sets_nowin_par4_35-80m_1000-3000d/tset_tot-%d_freq-%d_dynesty1' % (r*r,ndata), #location of test set for all plots ecept kl
        add_noise_real=True,                  # whether or not to add extra noise realizations in training set
        n_noise=n_noise,                      # number of noise realizations
        ref_gps_time=1126259643.0,            # reference gps time + 0.5s (t0 where test samples are injected+0.5s)
        do_normscale=True,                    # if true normalize parameters
        do_mc_eta_conversion=False,           # if True, convert m1 and m2 parameters into mc and eta
        n_kl_samp=100,                        # number of iterations in statistic tests
        do_adkskl_test=True,                  # if True, do statistic tests
        do_m1_m2_cut=False,                   # if True, make a cut on all m1 and m2 values    
        do_extra_noise=True,                  # add extra noise realizations during training
        do_load_in_chunks=False,              # if True, load training samples in random file chucnks every 25000 epochs
        Npp = 100,                             # number of test signals per pp-plot
        samplers=['vitamin','dynesty','emcee'],          # list of available bilby samplers to use
        use_samplers = [0,1,2],                  # number of Bilby samplers to use 
        kl_set_dir='test_sets_nowin_par4_35-80m_1000-3000d/tset_tot-%d_freq-%d' % (r*r,ndata) # location of test set used for kl
    )
    return params

# Bilby PE generation scripts
def tukey(M,alpha=0.5):
    """ Tukey window code copied from scipy.
    Parameters
    ----------
    M:
        Number of points in the output window.
    alpha:
        The fraction of the window inside the cosine tapered region.
    Returns
    -------
    w:
        The window
    """
    n = np.arange(0, M)
    width = int(np.floor(alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
    w = np.concatenate((w1, w2, w3))

    return np.array(w[:M])

def make_bbh(hp,hc,fs,ra,dec,psi,det,ifos,event_time):
    """ Turns hplus and hcross into a detector output
    applies antenna response and
    and applies correct time delays to each detector
    Parameters
    ----------
    hp:
        h-plus version of GW waveform
    hc:
        h-cross version of GW waveform
    fs:
        sampling frequency
    ra:
        right ascension
    dec:
        declination
    psi:
        polarization angle        
    det:
        detector
    Returns
    -------
    ht:
        combined h-plus and h-cross version of waveform
    hp:
        h-plus version of GW waveform 
    hc:
        h-cross version of GW waveform
    """
    # compute antenna response and apply
    ht = hp + hc     # overwrite the timeseries vector to reuse it

    return ht, hp, hc

def gen_template(duration,sampling_frequency,pars,ref_geocent_time,wvf_est=False):
    # whiten signal

    # fix parameters here
    if not wvf_est:
        injection_parameters = dict(
        mass_1=pars['m1'],mass_2=pars['m2'], a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
        phi_12=0.0, phi_jl=0.0, luminosity_distance=pars['lum_dist'], theta_jn=pars['theta_jn'], psi=pars['psi'],
        phase=pars['phase'], geocent_time=pars['geocent_time'], ra=pars['ra'], dec=pars['dec'])

    if wvf_est:
        injection_parameters = dict(
        mass_1=pars['m1'], mass_2=pars['m2'], a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
        phi_12=0.0, phi_jl=0.0, luminosity_distance=pars['lum_dist'], theta_jn=pars['theta_jn'], psi=pars['psi'],
        phase=pars['phase'], geocent_time=pars['geocent_time'], ra=pars['ra'], dec=pars['dec'])
    # Fixed arguments passed into the source model
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=20., minimum_frequency=20.)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
        start_time=ref_geocent_time-0.5)
    
    # create waveform
    wfg = waveform_generator
    # manually add time shifting in waveform generation
    wfg.parameters = injection_parameters
    freq_signal = wfg.frequency_domain_strain()
    time_signal = wfg.time_domain_strain()

    # Set up interferometers.  In this case we'll use two interferometers
    # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
    # sensitivity
    ifos = bilby.gw.detector.InterferometerList([pars['det']])

    # set noise to be colored Gaussian noise
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=ref_geocent_time-0.5)# - 0.5)

    # inject signal
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection_parameters)

    # get signal + noise
    signal_noise = ifos[0].strain_data.frequency_domain_strain

    # get time shifted noise-free signal
    freq_signal = ifos[0].get_detector_response(freq_signal, injection_parameters) 

    # whiten noise-free signal
    whiten_ht = freq_signal/ifos[0].amplitude_spectral_density_array

    # make aggressive window to cut out signal in central region
    # window is non-flat for 1/8 of desired Tobs
    # the window has dropped to 50% at the Tobs boundaries
    N = int(duration*sampling_frequency/2) + 1
    safe = 1.0                       # define the safe multiplication scale for the desired time length
    win = np.zeros(N)
    win = tukey(int(N/safe),alpha=1.0/16.0)
    #win[int((N-tempwin.size)):int((N-tempwin.size))+tempwin.size] = tempwin

    # apply aggressive window to cut out signal in central region
    # window is non-flat for 1/8 of desired Tobs
    # the window has dropped to 50% at the Tobs boundaries
    whiten_ht=whiten_ht.reshape(whiten_ht.shape[0])
    #whiten_ht[:] *= win

    ht = np.fft.irfft(whiten_ht)

    # noisy signal
    white_noise_sig = signal_noise/(ifos[0].amplitude_spectral_density_array)
    white_noise_sig = np.fft.irfft(white_noise_sig)

    # combine noise and noise-free signal
    ht_noisy = white_noise_sig 

    return ht,ht_noisy,injection_parameters,ifos,waveform_generator

def gen_masses(m_min=5.0,M_max=100.0,mdist='metric'):
    """ function returns a pair of masses drawn from the appropriate distribution
   
    Parameters
    ----------
    m_min:
        minimum component mass
    M_max:
        maximum total mass
    mdist:
        mass distribution to use when generating templates
    Returns
    -------
    m12: list
        both component mass parameters
    eta:
        eta parameter
    mc:
        chirp mass parameter
    """
    
    flag = False

    if mdist=='equal_mass':
        print('{}: using uniform and equal mass distribution'.format(time.asctime()))
        m1 = np.random.uniform(low=35.0,high=50.0)
        m12 = np.array([m1,m1])
        eta = m12[0]*m12[1]/(m12[0]+m12[1])**2
        mc = np.sum(m12)*eta**(3.0/5.0)
        return m12, mc, eta
    elif mdist=='uniform':
        print('{}: using uniform mass and non-equal mass distribution'.format(time.asctime()))
        new_m_min = m_min
        new_M_max = M_max
        while not flag:
            m1 = np.random.uniform(low=35.0,high=80.0)
            m2 = np.random.uniform(low=35.0,high=80.0)
            m12 = np.array([m1,m2]) 
            flag = True if (np.sum(m12)<new_M_max) and (np.all(m12>new_m_min)) and (m12[0]>=m12[1]) else False
        eta = m12[0]*m12[1]/(m12[0]+m12[1])**2
        mc = np.sum(m12)*eta**(3.0/5.0)
        return m12, mc, eta

    elif mdist=='astro':
        print('{}: using astrophysical logarithmic mass distribution'.format(time.asctime()))
        new_m_min = m_min
        new_M_max = M_max
        log_m_max = np.log(new_M_max - new_m_min)
        while not flag:
            m12 = np.exp(np.log(new_m_min) + np.random.uniform(0,1,2)*(log_m_max-np.log(new_m_min)))
            flag = True if (np.sum(m12)<new_M_max) and (np.all(m12>new_m_min)) and (m12[0]>=m12[1]) else False
        eta = m12[0]*m12[1]/(m12[0]+m12[1])**2
        mc = np.sum(m12)*eta**(3.0/5.0)
        return m12, mc, eta
    elif mdist=='metric':
        print('{}: using metric based mass distribution'.format(time.asctime()))
        new_m_min = m_min
        new_M_max = M_max
        new_M_min = 2.0*new_m_min
        eta_min = m_min*(new_M_max-new_m_min)/new_M_max**2
        while not flag:
            M = (new_M_min**(-7.0/3.0) - np.random.uniform(0,1,1)*(new_M_min**(-7.0/3.0) - new_M_max**(-7.0/3.0)))**(-3.0/7.0)
            eta = (eta_min**(-2.0) - np.random.uniform(0,1,1)*(eta_min**(-2.0) - 16.0))**(-1.0/2.0)
            m12 = np.zeros(2)
            m12[0] = 0.5*M + M*np.sqrt(0.25-eta)
            m12[1] = M - m12[0]
            flag = True if (np.sum(m12)<new_M_max) and (np.all(m12>new_m_min)) and (m12[0]>=m12[1]) else False
        mc = np.sum(m12)*eta**(3.0/5.0)
        return m12, mc, eta

def gen_par(fs,T_obs,geocent_time,mdist='metric'):
    """ Generates a random set of parameters
    
    Parameters
    ----------
    fs:
        sampling frequency (Hz)
    T_obs:
        observation time window (seconds)
    mdist:
        distribution of masses to use
    beta:
        fractional allowed window to place time series
    gw_tmp:
        if True: generate an event-like template
    Returns
    -------
    par: class object
        class containing parameters of waveform
    """
    # define distribution params
    m_min = 35.0         # 5 rest frame component masses
    M_max = 160.0       # 100 rest frame total mass

    m12, mc, eta = gen_masses(m_min,M_max,mdist=mdist)
    M = np.sum(m12)
    print('{}: selected bbh masses = {},{} (chirp mass = {})'.format(time.asctime(),m12[0],m12[1],mc))

    # generate reference phase
    # TODO: Need to change this back to 2*np.pi eventually
    phase = np.random.uniform(low=0.0,high=2*np.pi)
    #phase = 0.0
    print('{}: selected bbh reference phase = {}'.format(time.asctime(),phase))
    # generate reference inclination angle
    #theta_jn = np.random.uniform(low=0.0, high=2.0*np.pi)
    #print('{}: selected bbh inc angle = {}'.format(time.asctime(),theta_jn))

    geocent_time = np.random.uniform(low=geocent_time-0.1,high=geocent_time+0.1)
    print('{}: selected bbh GPS time = {}'.format(time.asctime(),geocent_time))

    lum_dist = np.random.uniform(low=1e3, high=3e3)
    #lum_dist = int(2e3)
    print('{}: selected bbh luminosity distance = {}'.format(time.asctime(),lum_dist))

    #theta_jn = np.random.uniform(low=0, high=np.pi)
    theta_jn = 0.0
    print('{}: selected bbh inclination angle = {}'.format(time.asctime(),theta_jn))

    return m12[0], m12[1], mc, eta, phase, geocent_time, lum_dist, theta_jn 

def run(sampling_frequency=512.,cnt=1.0,pos_test=[],file_test='',duration=1.,m1=36.,m2=36.,mc=17.41,
           geocent_time=1126259642.5,lum_dist=2000.,phase=0.0,N_gen=1000,make_test_samp=False,
           make_train_samp=False,run_label='test_results',make_noise=False,n_noise=25,outdir='bilby_output'):
    # Set the duration and sampling frequency of the data segment that we're
    # going to inject the signal into
    duration = duration
    sampling_frequency = sampling_frequency
    det='H1'
    ra=1.375
    dec=-1.2108
    psi=0.0
    theta_jn=0.0
    lum_dist=lum_dist
    mc=0
    eta=0
    ref_geocent_time=1126259642.5 # reference gps time

    pars = {'mc':mc,'geocent_time':geocent_time,'phase':phase,
            'N_gen':N_gen,'det':det,'ra':ra,'dec':dec,'psi':psi,'theta_jn':theta_jn,'lum_dist':lum_dist}

    # Specify the output directory and the name of the simulation.
    label = run_label
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    # We are going to inject a binary black hole waveform.  We first establish a
    # dictionary of parameters that includes all of the different waveform
    # parameters, including masses of the two black holes (mass_1, mass_2),
    # spins of both black holes (a, tilt, phi), etc.

    # generate training samples
    if make_train_samp == True:
        train_samples_noisefree = np.zeros((N_gen,sampling_frequency))
        train_samples_noisy = np.zeros((N_gen,sampling_frequency))
        train_pars = np.zeros((N_gen,6))
        for i in range(N_gen):
            # choose waveform parameters here
            pars['m1'], pars['m2'], mc, eta, pars['phase'], pars['geocent_time'], pars['lum_dist'], pars['theta_jn'] = gen_par(duration,sampling_frequency,geocent_time,mdist='uniform')
            if not make_noise:
                train_samples.append(gen_template(duration,sampling_frequency,
                                     pars,ref_geocent_time)[0:2])
                train_pars.append([pars['m1'],pars['phase'],pars['geocent_time'],pars['m2'],pars['lum_dist'],pars['theta_jn']])
                print('Made waveform %d/%d' % (i,N_gen))
            # generate extra noise realizations if requested
            if make_noise:
                for j in range(n_noise):
                    train_samples = np.array(gen_template(duration,sampling_frequency,
                                         pars,ref_geocent_time)[0:2])
                    train_samples_noisefree[i,:] = train_samples[0,:]
                    train_samples_noisy[i,:] = train_samples[1,:]
                    tmp_pars = np.asarray([pars['m1'],pars['phase'],pars['geocent_time'],pars['m2'],pars['lum_dist'],pars['theta_jn']]).reshape(6)
                    train_pars[i,:] = tmp_pars
                    print('Made unique waveform %d/%d' % (i,N_gen))
        return train_samples_noisy,train_samples_noisefree,np.array(train_pars)

    # generate testing sample 
    opt_snr = 0
    if make_test_samp == True:
        # ensure that signal is loud enough (e.g. > detection threshold)
        while opt_snr < 8:
            # generate parameters
            pars['m1'], pars['m2'], mc, eta, pars['phase'], pars['geocent_time'], pars['lum_dist'], pars['theta_jn']=gen_par(duration,sampling_frequency,geocent_time,mdist='uniform')
            # make gps time to be same as ref time
            pars['geocent_time']=ref_geocent_time

            # GW150914 parameters
            # uncomment to make randomized samples
            #pars['lum_dist'] = 410.0
            #pars['m1'] = 35.8 # source frame mass
            #pars['m2'] = 29.1 # source frame mass

            # inject signal
            test_samp_noisefree,test_samp_noisy,injection_parameters,ifos,waveform_generator = gen_template(duration,sampling_frequency,
                                   pars,ref_geocent_time)

            opt_snr = ifos[0].meta_data['optimal_SNR']
            print(ifos[0].meta_data['optimal_SNR'])

    # Set up a PriorDict, which inherits from dict.
    # By default we will sample all terms in the signal models.  However, this will
    # take a long time for the calculation, so for this example we will set almost
    # all of the priors to be equall to their injected values.  This implies the
    # prior is a delta function at the true, injected value.  In reality, the
    # sampler implementation is smart enough to not sample any parameter that has
    # a delta-function prior.
    # The above list does *not* include mass_1, mass_2, theta_jn and luminosity
    # distance, which means those are the parameters that will be included in the
    # sampler.  If we do nothing, then the default priors get used.
    priors = bilby.gw.prior.BBHPriorDict()
    priors['geocent_time'] = bilby.core.prior.Uniform(
        minimum=injection_parameters['geocent_time'] - 0.1,#duration/2,
        maximum=injection_parameters['geocent_time'] + 0.1,#duration/2,
        name='geocent_time', latex_label='$t_c$', unit='$s$')
    # fix the following parameter priors
    priors['mass_1'] = bilby.gw.prior.Uniform(name='mass_1', minimum=35.0, maximum=80.0,unit='$M_{\odot}$')
    priors['mass_2'] = bilby.gw.prior.Uniform(name='mass_2', minimum=35.0, maximum=80.0,unit='$M_{\odot}$')
    priors['a_1'] = 0
    priors['a_2'] = 0
    priors['tilt_1'] = 0
    priors['tilt_2'] = 0
    priors['phi_12'] = 0
    priors['phi_jl'] = 0
    priors['ra'] = 1.375
    priors['dec'] = -1.2108
    priors['psi'] = 0.0
    #priors['theta_jn'] = bilby.gw.prior.Uniform(name='theta_jn', minimum=0.0, maximum=np.pi)
    priors['theta_jn'] = 0.0 # range -1 to 1
    #priors['mass_ratio'] = 1.0
    #priors['chirp_mass'] = bilby.gw.prior.Uniform(name='chirp_mass', minimum=30.469269715364344, maximum=43.527528164806206, latex_label='$mc$', unit='$M_{\\odot}$')
    priors['phase'] = bilby.gw.prior.Uniform(name='phase', minimum=0.0, maximum=2*np.pi)
    #priors['phase'] = 0.0
    priors['luminosity_distance'] =  bilby.gw.prior.Uniform(name='luminosity_distance', minimum=1e3, maximum=3e3, unit='Mpc')
    #priors['luminosity_distance'] = int(2e3)

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveform generator
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator, phase_marginalization=False,
        priors=priors)

    # Run sampler.  In this case we're going to use the `dynesty` sampler
    # dynesty sampler
    #result = bilby.run_sampler(#conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    #    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
    #    injection_parameters=injection_parameters, outdir=outdir, label=label,
    #    save='hdf5')

    # emcee sampler
    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='emcee',
        nwalkers=100, nsteps=1000, nburn=500,
        injection_parameters=injection_parameters, outdir=outdir, label=label,
        save='hdf5')

    # Make a corner plot.
    result.plot_corner(parameters=['mass_1','mass_2','phase','geocent_time','luminosity_distance','theta_jn'])

    # save test sample waveform
    hf = h5py.File('%s/%s.h5py' % (outdir,run_label), 'w')
    hf.create_dataset('noisy_waveform', data=test_samp_noisy)
    hf.create_dataset('noisefree_waveform', data=test_samp_noisefree)
    hf.create_dataset('mass_1_post', data=np.array(result.posterior.mass_1))
    hf.create_dataset('mass_2_post', data=np.array(result.posterior.mass_2))
    #hf.create_dataset('mc_post', data=np.array(result.posterior.chirp_mass))
    hf.create_dataset('geocent_time_post', data=np.array(result.posterior.geocent_time))
    hf.create_dataset('luminosity_distance_post', data=np.array(result.posterior.luminosity_distance))
    hf.create_dataset('phase_post', data=np.array(result.posterior.phase))
    #hf.create_dataset('theta_jn_post', data=np.array(result.posterior.theta_jn))


    hf.create_dataset('mass_1', data=pars['m1'])
    hf.create_dataset('mass_2', data=pars['m2'])
    #hf.create_dataset('mc', data=mc)
    hf.create_dataset('geocent_time', data=result.injection_parameters['geocent_time'])
    hf.create_dataset('luminosity_distance', data=result.injection_parameters['luminosity_distance'])
    #hf.create_dataset('theta_jn', data=result.injection_parameters['theta_jn'])
    hf.create_dataset('phase', data=result.injection_parameters['phase'])
    
    hf.close()

    print('finished running pe')

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
        signal_train_images, sig, signal_train_pars = run(sampling_frequency=params['ndata'],N_gen=params['tset_split'],make_train_samp=True,make_test_samp=False,make_noise=params['add_noise_real'],n_noise=params['n_noise'])

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
        run(run_label='test_samp_%d' % i,make_test_samp=True,make_train_samp=False,duration=params['T'],sampling_frequency=params['ndata'],outdir=params['test_set_dir'])

# load test samples
pos_test, labels_test, sig_test = [], [], []
samples = np.zeros((params['r']*params['r'],params['n_samples'],params['ndim_x']+1))
cnt=0
for i in range(params['r']):
    for j in range(params['r']):
        # TODO: remove this bandaged phase file calc
        f = h5py.File('%s/test_samp_%d.h5py' % (params['test_set_dir'],cnt), 'r+')

        # select samples from posterior randomly
        shuffling = np.random.permutation(f['phase_post'][:].shape[0])
        phase = f['phase_post'][:][shuffling]
         
        if params['do_mc_eta_conversion']:
            m1 = f['mass_1_post'][:][shuffling]
            m2 = f['mass_2_post'][:][shuffling]
            eta = (m1*m2)/(m1+m2)**2
            mc = np.sum([m1,m2], axis=0)*eta**(3.0/5.0)
        else: 
            m1 = f['mass_1_post'][:][shuffling]
            m2 = f['mass_2_post'][:][shuffling]
        t0 = params['ref_gps_time'] - f['geocent_time_post'][:][shuffling]
        dist=f['luminosity_distance_post'][:][shuffling]
        #theta_jn=f['theta_jn_post'][:][shuffling]
        if params['do_mc_eta_conversion']:
            f_new=np.array([mc,phase,t0,eta]).T
        else:
            f_new=np.array([m1,phase,t0,m2,dist]).T
        f_new=f_new[:params['n_samples'],:]
        samples[cnt,:,:]=f_new

        # get true scalar parameters
        if params['do_mc_eta_conversion']:
            m1 = np.array(f['mass_1'])
            m2 = np.array(f['mass_2'])
            eta = (m1*m2)/(m1+m2)**2
            mc = np.sum([m1,m2])*eta**(3.0/5.0)
            pos_test.append([mc,np.array(f['phase']),params['ref_gps_time']-np.array(f['geocent_time']),eta])
        else:
            m1 = np.array(f['mass_1'])
            m2 = np.array(f['mass_2'])
            pos_test.append([m1,np.array(f['phase']),params['ref_gps_time']-np.array(f['geocent_time']),m2,np.array(f['luminosity_distance'])])
        labels_test.append([np.array(f['noisy_waveform'])])
        sig_test.append([np.array(f['noisefree_waveform'])])
        cnt += 1
        f.close()

pos_test = np.array(pos_test)
y_data_test_h = np.array(labels_test).reshape(int(r*r),ndata) * np.sqrt(2*params['ndata'])
sig_test = np.array(sig_test).reshape(int(r*r),ndata) * np.sqrt(2*params['ndata'])# noise-free y_test

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

        normscales = [np.max(data['x_data_train_h'][:,0]),np.max(data['x_data_train_h'][:,1]),np.max(data['x_data_train_h'][:,2]),np.max(data['x_data_train_h'][:,3]),np.max(data['x_data_train_h'][:,4])]#,np.max(data['pos'][:,5])]

        data['x_data_train_h'][:,0]=data['x_data_train_h'][:,0]/normscales[0]
        data['x_data_train_h'][:,1]=data['x_data_train_h'][:,1]/normscales[1]
        data['x_data_train_h'][:,2]=data['x_data_train_h'][:,2]/normscales[2]
        data['x_data_train_h'][:,3]=data['x_data_train_h'][:,3]/normscales[3]
        data['x_data_train_h'][:,4]=data['x_data_train_h'][:,4]/normscales[4]
    #    data['x_data_train_h'][:,5]=data['x_data_train_h'][:,5]/normscales[5]

    x_data_train_h = data['x_data_train_h']
    y_data_train_lh = data['y_data_train_lh'] * np.sqrt(2*params['ndata'])
    y_data_train_noisefree = data['y_data_train_noisefree'] * np.sqrt(2*params['ndata'])

    if params['do_normscale']:
        y_normscale = [np.max(y_data_train_lh)] # [1]
        y_data_train_lh=y_data_train_lh/y_normscale[0]
        y_data_test_h = y_data_test_h / y_normscale[0]

#    y_data_train_noisefree = 0
    # Remove phase parameter
    pos_test = pos_test[:,[0,2,3,4]]
    x_data_train_h = x_data_train_h[:,[0,2,3,4]]
    samples = samples[:,:,[0,2,3,4]]

    if params['do_normscale']: normscales = [normscales[0],normscales[2],normscales[3],normscales[4]]#,normscales[5]]
    x_data_train, y_data_train_l, y_data_train_h = x_data_train_h, y_data_train_lh, y_data_train_lh

if params['do_load_in_chunks']:

    if params['do_normscale']:
        normscales = [80.0,2*np.pi,0.6,80.0,3000.0]#,np.max(data['pos'][:,5])]
        normscales = [normscales[0],normscales[2],normscales[3],normscales[4]]#,normscales[5]]

    # Remove phase parameter
    pos_test = pos_test[:,[0,2,3,4]]
    samples = samples[:,:,[0,2,3,4]]
    y_data_train_noisefree = 0 * np.sqrt(2*params['ndata'])

# Make directory for plots
#plots.make_dirs(params['plot_dir'][0])
# Declare plot class variables
plotter = plots.make_plots(params,samples,None,pos_test)

#loss_inv=[0.0]
#kl_inv=[0.0]

olvec = np.zeros((params['r'],params['r'],int(iterations/(params['num_iterations']-1))))
s=0
olvec_2d = np.zeros((params['r'],params['r'],int(iterations/params['num_iterations']),6))
adksVec = np.zeros((params['r'],params['r'],params['ndim_x'],6,params['n_kl_samp']))

"""
if params['do_load_in_chunks']:
    # Get first set of training samples
    x_data_train, y_data_train_l, y_data_train_h, x_data_train_h, y_data_train_lh =  chris_data.load_training_set(params,train_files,normscales)
    y_data_train_h, y_data_train_lh = y_data_train_h * np.sqrt(2*params['ndata']), y_data_train_lh * np.sqrt(2*params['ndata'])

for i in range(int(iterations/(params['num_iterations']-1))):
    print('iteration %d' % (i*int(params['num_iterations']-1)))

    if i == 0:
        loss_inv, kl_inv, train_files = VICI_inverse_model.train(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'], plotter, y_data_test_h,train_files,normscales,y_data_train_noisefree,y_normscale) # This trains the inverse model to recover posteriors using the forward model weights stored in forward_model_dir/forward_model.ckpt and saves the inverse model weights in inverse_model_dir/inverse_model.ckpt
        #loss_inv, kl_inv = VICI_inverse_model.resume_training(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'],train_files,normscales,y_data_train_noisefree)

        olvec = np.zeros((params['r'],params['r'],int(iterations/(params['num_iterations']-1))))
        s=0
    else:
        loss_temp, kl_temp, train_files = VICI_inverse_model.resume_training(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'],train_files,normscales,y_data_train_noisefree,y_normscale)

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

        # rescale t0 back to its nominal values
#        XS[:,1,:] -= 100.5


        # Generate final results plots
        plotter = plots.make_plots(params,samples,XS,pos_test)

        # Geneerate overlap scatter plots
        s,olvec,olvec_2d = plotter.make_overlap_plot((i*int(params['num_iterations']-1)),iterations,s,olvec,olvec_2d,adksVec)

"""


# The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
xm, xsx, XS, pmax = VICI_inverse_model.run(params, y_data_test_h, np.shape(x_data_train)[1], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label']) # This runs the trained model using the weights stored in inverse_model_dir/inverse_model.ckpt
# The outputs are the following:
# - xm = marginal means
# - xsx = marginal standard deviations
# - XS = draws from the posterior (3D array with different samples for the same input along the third dimension)
# - pmax = approximate maxima (approximate 'best' reconstructions)

# Convert XS back to unnormalized version
if params['do_normscale']:
    for m in range(params['ndim_x']):
        XS[:,m,:] = XS[:,m,:]*normscales[m]

# Generate final results plots
plotter = plots.make_plots(params,samples,XS,pos_test)

# Make KL plot
plotter.gen_kl_plots(VICI_inverse_model,y_data_test_h,x_data_train,normscales)

# Make pp plot
#plotter.plot_pp(VICI_inverse_model,y_data_train_l,x_data_train,0,normscales)

# Geneerate overlap scatter plots
plotter.make_overlap_plot(0,iterations,s,olvec,olvec_2d,adksVec)

