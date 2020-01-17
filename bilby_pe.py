#! /usr/bin/env python

"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""

from __future__ import division, print_function

import numpy as np
import bilby
from sys import exit
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import scipy
import lalsimulation
import lal
import time
import h5py
from scipy.ndimage.interpolation import shift
#from pylal import antenna, cosmography
import argparse

def parser():
    """
    Parses command line arguments
    :return: arguments
    """

    #TODO: complete help sections
    parser = argparse.ArgumentParser(prog='bilby_pe.py', description='script for generating bilby samples/posterior')

    # arguments for data
    parser.add_argument('-label', '--run-label', type=str, default='1',
                        help='label that identifies which bilby results this is')
    parser.add_argument('-od', '--outdir', type=str, default='test',
                        help='Directory to store bilby results')

    return parser.parse_args()

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
    #Fp=ifos.antenna_response(ra,dec,float(event_time),psi,'plus')
    #Fc=ifos.antenna_response(ra,dec,float(event_time),psi,'cross')
    #Fp,Fc,_,_ = antenna.response(float(event_time), ra, dec, 0, psi, 'radians', det )
    ht = hp + hc     # overwrite the timeseries vector to reuse it

    return ht, hp, hc

def gen_template(duration,
                 sampling_frequency,
                 pars,
                 ref_geocent_time
                 ):
    """
    Generates a whitened waveform
    """

    if sampling_frequency>4096:
        print('EXITING: bilby doesn\'t seem to generate noise above 2048Hz so lower the sampling frequency')
        exit(0)

    # compute the number of time domain samples
    Nt = int(sampling_frequency*duration)

    # define the start time of the timeseries
    start_time = ref_geocent_time-duration/2.0

    # fix parameters here
    injection_parameters = dict(
        mass_1=pars['mass_1'],mass_2=pars['mass_2'], a_1=pars['a_1'], a_2=pars['a_2'], tilt_1=pars['tilt_1'], tilt_2=pars['tilt_2'],
        phi_12=pars['phi_12'], phi_jl=pars['phi_jl'], luminosity_distance=pars['luminosity_distance'], theta_jn=pars['theta_jn'], psi=pars['psi'],
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
        start_time=start_time)
    
    # create waveform
    wfg = waveform_generator

    # extract waveform from bilby
    wfg.parameters = injection_parameters
    freq_signal = wfg.frequency_domain_strain()
    time_signal = wfg.time_domain_strain()

    # Set up interferometers. These default to their design
    # sensitivity
    ifos = bilby.gw.detector.InterferometerList([pars['det']])

    # set noise to be colored Gaussian noise
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=start_time)

    # inject signal
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection_parameters)

    # get frequency domain noise-free signal at detector
    signal_fd = ifos[0].get_detector_response(freq_signal, injection_parameters) 

    # whiten frequency domain noise-free signal (and reshape/flatten)
    whitened_signal_fd = signal_fd/ifos[0].amplitude_spectral_density_array
    #whitened_signal_fd = whitened_signal_fd.reshape(whitened_signal_fd.shape[0])    

    # get frequency domain signal + noise at detector
    h_fd = ifos[0].strain_data.frequency_domain_strain

    # inverse FFT noise-free signal back to time domain and normalise
    whitened_signal_td = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_signal_fd)

    # whiten noisy frequency domain signal
    whitened_h_fd = h_fd/ifos[0].amplitude_spectral_density_array
    
    # inverse FFT noisy signal back to time domain and normalise
    whitened_h_td = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_h_fd)

    return whitened_signal_td,whitened_h_td,injection_parameters,ifos,waveform_generator

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
        return m12[0], m12[1], np.sum(m12), mc, eta
    elif mdist=='uniform':
        print('{}: using uniform mass and non-equal mass distribution'.format(time.asctime()))
        new_m_min = m_min
        new_M_max = M_max
        while not flag:
            m1 = np.random.uniform(low=new_m_min,high=M_max/2.0)
            m2 = np.random.uniform(low=new_m_min,high=M_max/2.0)
            m12 = np.array([m1,m2]) 
            flag = True if (np.sum(m12)<new_M_max) and (np.all(m12>new_m_min)) and (m12[0]>=m12[1]) else False
        eta = m12[0]*m12[1]/(m12[0]+m12[1])**2
        mc = np.sum(m12)*eta**(3.0/5.0)
        return m12[0], m12[1], np.sum(m12), mc, eta

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
        return m12[0], m12[1], np.sum(m12), mc, eta
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
        return m12[0], m12[1], np.sum(m12), mc, eta

def gen_par(pars,
            rand_pars=[None],
            bounds=None,
            mdist='uniform'
            ):
    """ 
    Generates a random set of parameters
    """

    # make masses
    if np.any([r=='mass_1' for r in rand_pars]):
        pars['mass_1'], pars['mass_2'], pars['M'], pars['mc'], pars['eta'] = gen_masses(bounds['mass_1_min'],bounds['M_max'],mdist=mdist)
        print('{}: selected bbh masses = {},{} (chirp mass = {})'.format(time.asctime(),pars['mass_1'],pars['mass_2'],pars['mc']))

    # generate reference phase
    if np.any([r=='phase' for r in rand_pars]):
        pars['phase'] = np.random.uniform(low=bounds['phase_min'],high=bounds['phase_max'])
        print('{}: selected bbh reference phase = {}'.format(time.asctime(),pars['phase']))

    # generate polarisation
    if np.any([r=='psi' for r in rand_pars]):
        pars['psi'] = np.random.uniform(low=bounds['psi_min'],high=bounds['psi_max'])
        print('{}: selected bbh polarisation = {}'.format(time.asctime(),pars['psi']))

    # generate RA
    if np.any([r=='ra' for r in rand_pars]):
        pars['ra'] = np.random.uniform(low=bounds['ra_min'],high=bounds['ra_max'])
        print('{}: selected bbh right ascension = {}'.format(time.asctime(),pars['ra']))

    # generate declination
    if np.any([r=='dec' for r in rand_pars]):
        pars['dec'] = np.arcsin(np.random.uniform(low=np.sin(bounds['phase_min']),high=np.sin(bounds['dec_max'])))
        print('{}: selected bbh declination = {}'.format(time.asctime(),pars['dec']))

    # make geocentric arrival time
    if np.any([r=='geocent_time' for r in rand_pars]):
        pars['geocent_time'] = np.random.uniform(low=bounds['geocent_time_min'],high=bounds['geocent_time_max'])
        print('{}: selected bbh GPS time = {}'.format(time.asctime(),pars['geocent_time']))

    # make distance
    if np.any([r=='luminosity_distance' for r in rand_pars]):
        pars['luminosity_distance'] = np.random.uniform(low=bounds['luminosity_distance_min'], high=bounds['luminosity_distance_max'])
        print('{}: selected bbh luminosity distance = {}'.format(time.asctime(),pars['luminosity_distance']))

    # make inclination
    if np.any([r=='theta_jn' for r in rand_pars]):
        pars['theta_jn'] = np.arccos(np.random.uniform(low=np.cos(bounds['theta_jn_min']),high=np.cos(bounds['theta_jn_max'])))
        print('{}: selected bbh inclination angle = {}'.format(time.asctime(),pars['theta_jn']))

    return pars

##########################################################################
def run(sampling_frequency=256.0,
           duration=1.,
           N_gen=1000,
           bounds=None,
           fixed_vals=None,
           rand_pars=[None],
           inf_pars=[None],
           ref_geocent_time=1126259642.5,
           training=True,
           do_pe=False,
           label='test_results',
           out_dir='bilby_output',
           seed=None,
           ):
    """
    Generate data sets
    """

    # Set up a random seed for result reproducibility.  This is optional!
    if seed is not None:
        np.random.seed(seed)

    # generate training samples
    if training == True:
        train_samples = []
        train_pars = []
        for i in range(N_gen):
            
            # choose waveform parameters here
            pars = gen_par(fixed_vals,bounds=bounds,rand_pars=rand_pars,mdist='uniform')
            
            # store the params
            temp = []
            for p in rand_pars:
                for q,qi in pars.items():
                    if p==q:
                        temp.append(qi)
            train_pars.append([temp])
        
            # make the data - shift geocent time to correct reference
            pars['geocent_time'] += ref_geocent_time
            train_samples.append(gen_template(duration,sampling_frequency,pars,ref_geocent_time)[0:2])
            print('Made waveform %d/%d' % (i,N_gen)) 

        train_samples_noisefree = np.array(train_samples)[:,0,:]
        train_samples_noisy = np.array(train_samples)[:,1,:]
        return train_samples_noisy,train_samples_noisefree,np.array(train_pars)

    # otherwise we are doing test data 
    else:
        
        # generate parameters
        pars = gen_par(fixed_vals,bounds=bounds,rand_pars=rand_pars,mdist='uniform')
        temp = []
        for p in rand_pars:
            for q,qi in pars.items():
                if p==q:
                    temp.append(qi)        

        # inject signal - shift geocent time to correct reference
        pars['geocent_time'] += ref_geocent_time
        test_samples_noisefree,test_samples_noisy,injection_parameters,ifos,waveform_generator = gen_template(duration,sampling_frequency,
                               pars,ref_geocent_time)

        # if not doing PE then return signal data
        if not do_pe:
            return test_samples_noisy,test_samples_noisefree,np.array([temp])

        bilby.core.utils.setup_logger(outdir=out_dir, label=label)

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
        if np.any([r=='geocent_time' for r in inf_pars]):
            priors['geocent_time'] = bilby.core.prior.Uniform(
                minimum=ref_geocent_time + bounds['geocent_time_min'],
                maximum=ref_geocent_time + bounds['geocent_time_max'],
                name='geocent_time', latex_label='$t_c$', unit='$s$')
        else:
            priors['geocent_time'] = ref_geocent_time + fixed_vals['geocent_time']
        if np.any([r=='mass_1' for r in inf_pars]):
            priors['mass_1'] = bilby.gw.prior.Uniform(name='mass_1', minimum=bounds['mass_1_min'], maximum=bounds['mass_1_max'],unit='$M_{\odot}$')
        else:
            priors['mass_1'] = fixed_vals['mass_1']
        if np.any([r=='mass_2' for r in inf_pars]):
            priors['mass_2'] = bilby.gw.prior.Uniform(name='mass_2', minimum=bounds['mass_2_min'], maximum=bounds['mass_2_max'],unit='$M_{\odot}$')
        else:
            priors['mass_2'] = fixed_vals['mass_2']
        if np.any([r=='a_1' for r in inf_pars]):
            priors['a_1'] = bilby.gw.prior.Uniform(name='a_1', minimum=bounds['a_1_min'], maximum=bounds['a_1_max'])
        else:
            priors['a_1'] = fixed_vals['a_1']
        if np.any([r=='a_2' for r in inf_pars]):
            priors['a_2'] = bilby.gw.prior.Uniform(name='a_2', minimum=bounds['a_2_min'], maximum=bounds['a_2_max'])
        else:
            priors['a_2'] = fixed_vals['a_2']
        if np.any([r=='tilt_1' for r in inf_pars]):
            priors['tilt_1'] = bilby.gw.prior.Uniform(name='tilt_1', minimum=bounds['tilt_1_min'], maximum=bounds['tilt_1_max'])
        else:
            priors['tilt_1'] = fixed_vals['tilt_1']
        if np.any([r=='tilt_2' for r in inf_pars]):
            priors['tilt_2'] = bilby.gw.prior.Uniform(name='tilt_2', minimum=bounds['tilt_2_min'], maximum=bounds['tilt_2_max'])
        else:
            priors['tilt_2'] = fixed_vals['tilt_2']
        if np.any([r=='phi_12' for r in inf_pars]):
            priors['phi_12'] = bilby.gw.prior.Uniform(name='phi_12', minimum=bounds['phi_12_min'], maximum=bounds['phi_12_max'])
        else:
            priors['phi_12'] = fixed_vals['phi_12']
        if np.any([r=='phi_jl' for r in inf_pars]):
            priors['phi_jl'] = bilby.gw.prior.Uniform(name='phi_jl', minimum=bounds['phi_jl_min'], maximum=bounds['phi_jl_max'])
        else:
            priors['phi_jl'] = fixed_vals['phi_jl']
        if np.any([r=='ra' for r in inf_pars]):
            priors['ra'] = bilby.gw.prior.Uniform(name='ra', minimum=bounds['ra_min'], maximum=bounds['ra_max'], boundary='periodic')
        else:
            priors['ra'] = fixed_vals['ra']
        if np.any([r=='dec' for r in inf_pars]):
            priors['dec'] = bilby.gw.prior.Cosine(name='dec', boundary='reflective')
        else:    
            priors['dec'] = fixed_vals['dec']
        if np.any([r=='psi' for r in inf_pars]):
            priors['psi'] = bilby.gw.prior.Uniform(name='psi', minimum=bounds['psi_min'], maximum=bounds['psi_max'], boundary='periodic')
        else:
            priors['psi'] = fixed_vals['psi']
        if np.any([r=='theta_jn' for r in inf_pars]):
#            priors['theta_jn'] = bilby.gw.prior.Sine(name='theta_jn', minimum=bounds['theta_jn_min'], maximum=bounds['theta_jn_max'], boundary='reflective')
             pass
        else:
            priors['theta_jn'] = fixed_vals['theta_jn']
        if np.any([r=='phase' for r in inf_pars]):
            priors['phase'] = bilby.gw.prior.Uniform(name='phase', minimum=bounds['phase_min'], maximum=bounds['phase_max'], boundary='periodic')
        else:
            priors['phase'] = fixed_vals['phase']
        if np.any([r=='luminosity_distance' for r in inf_pars]):
            priors['luminosity_distance'] =  bilby.gw.prior.Uniform(name='luminosity_distance', minimum=bounds['luminosity_distance_min'], maximum=bounds['luminosity_distance_max'], unit='Mpc')
        else:
            priors['luminosity_distance'] = fixed_vals['luminosity_distance']

        # Initialise the likelihood by passing in the interferometer data (ifos) and
        # the waveform generator
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator, phase_marginalization=False,
            priors=priors)

        run_startt = time.time()
        # Run sampler dynesty 1 sampler
        result = bilby.run_sampler(#conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            likelihood=likelihood, priors=priors, sampler='dynesty', npoints=5000,
            injection_parameters=injection_parameters, outdir=out_dir+'_dynesty1', label=label, dlogz=0.1,
            save='hdf5')
        run_endt = time.time()

        # save test sample waveform
        hf = h5py.File('%s/%s.h5py' % (out_dir+'_dynesty1',label), 'w')
        hf.create_dataset('noisy_waveform', data=test_samples_noisy)
        hf.create_dataset('noisefree_waveform', data=test_samples_noisefree)
        
        # loop over randomised params and save samples
        for p in inf_pars:
            for q,qi in result.posterior.items():
                if p==q:
                    name = p + '_post'
                    print('saving PE samples for parameter {}'.format(q))
                    hf.create_dataset(name, data=np.array(qi))
        hf.create_dataset('runtime', data=(run_endt - run_startt))
        hf.close()

        print('finished running pe')
        return test_samples_noisy,test_samples_noisefree,np.array([temp])

    #run_startt = time.time()
    # ptemcee sampler 1
    #result = bilby.run_sampler(
    #    likelihood=likelihood, priors=priors, sampler='ptemcee',
    #    nwalkers=100, nsteps=9000, nburn=8000, ntemps=2, 
    #    injection_parameters=injection_parameters, outdir=outdir+'_ptemcee1', label=label,
    #    save='hdf5')
    #run_endt = time.time()

    # Make a corner plot.
    #result.plot_corner(parameters=['mass_1','mass_2','phase','geocent_time','luminosity_distance','theta_jn'])

    # save test sample waveform
    #hf = h5py.File('%s/%s.h5py' % (outdir+'_ptemcee1',run_label), 'w')
    #hf.create_dataset('noisy_waveform', data=test_samp_noisy)
    #hf.create_dataset('noisefree_waveform', data=test_samp_noisefree)
    #hf.create_dataset('mass_1_post', data=np.array(result.posterior.mass_1))
    #hf.create_dataset('mass_2_post', data=np.array(result.posterior.mass_2))
    #hf.create_dataset('geocent_time_post', data=np.array(result.posterior.geocent_time))
    #hf.create_dataset('luminosity_distance_post', data=np.array(result.posterior.luminosity_distance))
    #hf.create_dataset('phase_post', data=np.array(result.posterior.phase))
    #hf.create_dataset('mass_1', data=pars['m1'])
    #hf.create_dataset('mass_2', data=pars['m2'])
    #hf.create_dataset('geocent_time', data=result.injection_parameters['geocent_time'])
    #hf.create_dataset('luminosity_distance', data=result.injection_parameters['luminosity_distance'])
    #hf.create_dataset('phase', data=result.injection_parameters['phase'])
    #hf.create_dataset('runtime', data=(run_endt - run_startt))
    #hf.close()

    #run_startt = time.time()
    # ptemcee sampler 2
    #result = bilby.run_sampler(
    #    likelihood=likelihood, priors=priors, sampler='ptemcee',
    #    nwalkers=100, nsteps=9000, nburn=8000, ntemps=2,
    #    injection_parameters=injection_parameters, outdir=outdir+'_ptemcee2', label=label,
    #    save='hdf5')
    #run_endt = time.time()

    # Make a corner plot.
    #result.plot_corner(parameters=['mass_1','mass_2','phase','geocent_time','luminosity_distance','theta_jn'])

    # save test sample waveform
    #hf = h5py.File('%s/%s.h5py' % (outdir+'_ptemcee2',run_label), 'w')
    #hf.create_dataset('noisy_waveform', data=test_samp_noisy)
    #hf.create_dataset('noisefree_waveform', data=test_samp_noisefree)
    #hf.create_dataset('mass_1_post', data=np.array(result.posterior.mass_1))
    #hf.create_dataset('mass_2_post', data=np.array(result.posterior.mass_2))
    #hf.create_dataset('geocent_time_post', data=np.array(result.posterior.geocent_time))
    #hf.create_dataset('luminosity_distance_post', data=np.array(result.posterior.luminosity_distance))
    #hf.create_dataset('phase_post', data=np.array(result.posterior.phase))
    #hf.create_dataset('mass_1', data=pars['m1'])
    #hf.create_dataset('mass_2', data=pars['m2'])
    #hf.create_dataset('geocent_time', data=result.injection_parameters['geocent_time'])
    #hf.create_dataset('luminosity_distance', data=result.injection_parameters['luminosity_distance'])
    #hf.create_dataset('phase', data=result.injection_parameters['phase'])
    #hf.create_dataset('runtime', data=(run_endt - run_startt))
    #hf.close()

    #run_startt = time.time()
    # Run dynesty 2 sampler.
    #result = bilby.run_sampler(#conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    #    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=5000,
    #    injection_parameters=injection_parameters, outdir=outdir+'_dynesty2', label=label, dlogz=0.1,
    #    save='hdf5')
    #run_endt = time.time()

    # save test sample waveform
    #hf = h5py.File('%s/%s.h5py' % (outdir+'_dynesty2',run_label), 'w')
    #hf.create_dataset('noisy_waveform', data=test_samp_noisy)
    #hf.create_dataset('noisefree_waveform', data=test_samp_noisefree)
    #hf.create_dataset('mass_1_post', data=np.array(result.posterior.mass_1))
    #hf.create_dataset('mass_2_post', data=np.array(result.posterior.mass_2))
    #hf.create_dataset('geocent_time_post', data=np.array(result.posterior.geocent_time))
    #hf.create_dataset('luminosity_distance_post', data=np.array(result.posterior.luminosity_distance))
    #hf.create_dataset('phase_post', data=np.array(result.posterior.phase))
    #hf.create_dataset('mass_1', data=pars['m1'])
    #hf.create_dataset('mass_2', data=pars['m2'])
    #hf.create_dataset('geocent_time', data=result.injection_parameters['geocent_time'])
    #hf.create_dataset('luminosity_distance', data=result.injection_parameters['luminosity_distance'])
    #hf.create_dataset('phase', data=result.injection_parameters['phase'])
    #hf.create_dataset('runtime', data=(run_endt - run_startt))
    #hf.close()

    #run_startt = time.time()
    # emcee sampler
    #result = bilby.run_sampler(
    #    likelihood=likelihood, priors=priors, sampler='emcee',
    #    nwalkers=100, nsteps=9000, nburn=8000,
    #    injection_parameters=injection_parameters, outdir=outdir+'_emcee1', label=label,
    #    save='hdf5')
    #run_endt = time.time()

    # Make a corner plot.
    #result.plot_corner(parameters=['mass_1','mass_2','phase','geocent_time','luminosity_distance','theta_jn'])

    # save test sample waveform
    #hf = h5py.File('%s/%s.h5py' % (outdir+'_emcee1',run_label), 'w')
    #hf.create_dataset('noisy_waveform', data=test_samp_noisy)
    #hf.create_dataset('noisefree_waveform', data=test_samp_noisefree)
    #hf.create_dataset('mass_1_post', data=np.array(result.posterior.mass_1))
    #hf.create_dataset('mass_2_post', data=np.array(result.posterior.mass_2))
    #hf.create_dataset('geocent_time_post', data=np.array(result.posterior.geocent_time))
    #hf.create_dataset('luminosity_distance_post', data=np.array(result.posterior.luminosity_distance))
    #hf.create_dataset('phase_post', data=np.array(result.posterior.phase))
    #hf.create_dataset('mass_1', data=pars['m1'])
    #hf.create_dataset('mass_2', data=pars['m2'])
    #hf.create_dataset('geocent_time', data=result.injection_parameters['geocent_time'])
    #hf.create_dataset('luminosity_distance', data=result.injection_parameters['luminosity_distance'])
    #hf.create_dataset('phase', data=result.injection_parameters['phase'])
    #hf.create_dataset('runtime', data=(run_endt - run_startt))
    #hf.close()

    
    #run_startt = time.time()
    # emcee sampler
    #result = bilby.run_sampler(
    #    likelihood=likelihood, priors=priors, sampler='emcee',
    #    nwalkers=100, nsteps=9000, nburn=8000,
    #    injection_parameters=injection_parameters, outdir=outdir+'_emcee2', label=label,
    #    save='hdf5')
    #run_endt = time.time()

    # Make a corner plot.
    #result.plot_corner(parameters=['mass_1','mass_2','phase','geocent_time','luminosity_distance','theta_jn'])

    # save test sample waveform
    #hf = h5py.File('%s/%s.h5py' % (outdir+'_emcee2',run_label), 'w')
    #hf.create_dataset('noisy_waveform', data=test_samp_noisy)
    #hf.create_dataset('noisefree_waveform', data=test_samp_noisefree)
    #hf.create_dataset('mass_1_post', data=np.array(result.posterior.mass_1))
    #hf.create_dataset('mass_2_post', data=np.array(result.posterior.mass_2))
    #hf.create_dataset('geocent_time_post', data=np.array(result.posterior.geocent_time))
    #hf.create_dataset('luminosity_distance_post', data=np.array(result.posterior.luminosity_distance))
    #hf.create_dataset('phase_post', data=np.array(result.posterior.phase))
    #hf.create_dataset('mass_1', data=pars['m1'])
    #hf.create_dataset('mass_2', data=pars['m2'])
    #hf.create_dataset('geocent_time', data=result.injection_parameters['geocent_time'])
    #hf.create_dataset('luminosity_distance', data=result.injection_parameters['luminosity_distance'])
    #hf.create_dataset('phase', data=result.injection_parameters['phase'])
    #hf.create_dataset('runtime', data=(run_endt - run_startt))
    #hf.close()
  
    
 
    #run_startt = time.time()
    # CP Nest sampler 1
    #result = bilby.run_sampler(
    #    likelihood=likelihood, priors=priors, sampler='cpnest',
    #    nlive=5000,maxmcmc=1000, seed=1994,
    #    injection_parameters=injection_parameters, outdir=outdir+'_cpnest1', label=label,
    #    save='hdf5')
    #run_endt = time.time()

    # Make a corner plot.
    #result.plot_corner(parameters=['mass_1','mass_2','phase','geocent_time','luminosity_distance'])

    # save test sample waveform
    #hf = h5py.File('%s/%s.h5py' % (outdir+'_cpnest1',run_label), 'w')
    #hf.create_dataset('noisy_waveform', data=test_samp_noisy)
    #hf.create_dataset('noisefree_waveform', data=test_samp_noisefree)
    #hf.create_dataset('mass_1_post', data=np.array(result.posterior.mass_1))
    #hf.create_dataset('mass_2_post', data=np.array(result.posterior.mass_2))
    #hf.create_dataset('geocent_time_post', data=np.array(result.posterior.geocent_time))
    #hf.create_dataset('luminosity_distance_post', data=np.array(result.posterior.luminosity_distance))
    #hf.create_dataset('phase_post', data=np.array(result.posterior.phase))
    #hf.create_dataset('mass_1', data=pars['m1'])
    #hf.create_dataset('mass_2', data=pars['m2'])
    #hf.create_dataset('geocent_time', data=result.injection_parameters['geocent_time'])
    #hf.create_dataset('luminosity_distance', data=result.injection_parameters['luminosity_distance'])
    #hf.create_dataset('phase', data=result.injection_parameters['phase'])
    #hf.create_dataset('runtime', data=(run_endt - run_startt))
    #hf.close()

    #run_startt = time.time()
    # CP Nest sampler 2
    #result = bilby.run_sampler(
    #    likelihood=likelihood, priors=priors, sampler='cpnest',
    #    nlive=5000,maxmcmc=1000, seed=42,
    #    injection_parameters=injection_parameters, outdir=outdir+'_cpnest2', label=label,
    #    save='hdf5')
    #run_endt = time.time()

    # Make a corner plot.
    #result.plot_corner(parameters=['mass_1','mass_2','phase','geocent_time','luminosity_distance'])

    # save test sample waveform
    #hf = h5py.File('%s/%s.h5py' % (outdir+'_cpnest2',run_label), 'w')
    #hf.create_dataset('noisy_waveform', data=test_samp_noisy)
    #hf.create_dataset('noisefree_waveform', data=test_samp_noisefree)
    #hf.create_dataset('mass_1_post', data=np.array(result.posterior.mass_1))
    #hf.create_dataset('mass_2_post', data=np.array(result.posterior.mass_2))
    #hf.create_dataset('geocent_time_post', data=np.array(result.posterior.geocent_time))
    #hf.create_dataset('luminosity_distance_post', data=np.array(result.posterior.luminosity_distance))
    #hf.create_dataset('phase_post', data=np.array(result.posterior.phase))
    #hf.create_dataset('mass_1', data=pars['m1'])
    #hf.create_dataset('mass_2', data=pars['m2'])
    #hf.create_dataset('geocent_time', data=result.injection_parameters['geocent_time'])
    #hf.create_dataset('luminosity_distance', data=result.injection_parameters['luminosity_distance'])
    #hf.create_dataset('phase', data=result.injection_parameters['phase'])
    #hf.create_dataset('runtime', data=(run_endt - run_startt))
    #hf.close() 
 
    print('finished running pe')

def main(args):
    run(run_label=args.run_label, outdir=args.outdir)

if __name__ == '__main__':
    args = parser()
    main(args)

