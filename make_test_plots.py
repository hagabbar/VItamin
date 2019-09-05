import numpy as np
import tensorflow as tf
import h5py

from Models import VICI_inverse_model
import plots

run_label='gpu4',            # label for run
plot_dir="/home/hunter.gabbard/public_html/CBC/VItamin/gw_results/%s" % run_label,                 # plot directory
ndata=256                    # y dimension size
load_test_set = True         # if True, load previously made test samples (including bilby posterior)
r = 16                        # the grid dimension for the output tests
n_noise=1                    # this is a redundant parameter. Needs to be removed TODO
ref_geocent_time=1126259642.5            # reference gps time

# Defining the list of parameter that need to be fed into the models
def get_params():
    params = dict(
        image_size = [1,ndata],       # Images Size
        print_values=True,            # optionally print values every report interval
        n_samples = 3000,             # number of posterior samples to save per reconstruction upon inference 

        ndata = ndata,
        r = r,                                # the grid dimension for the output tests
        ndim_x=4,                             # number of parameters to PE on
        sigma=1.0,                            # stadnard deviation of the noGise on signal
        usepars=[0,2,3,4],                    # which parameters you want to do PE on
        prior_min=[35.0,0.0,ref_geocent_time+0.15,35.0,1000.0],                         # minimum prior range
        prior_max=[80.0,2*np.pi,ref_geocent_time+0.35,80.0,3000.0],                         # maximum prior range
        seed=42,                              # random seed number
        run_label=run_label,                  # label for run
        plot_dir=plot_dir,                    # plot directory
        parnames=['m1','t0','m2','lum_dist'], # parameter names
        ref_geocent_time=ref_geocent_time,            # reference gps time 
        do_normscale=True,                    # if true normalize parameters
        do_mc_eta_conversion=False,           # if True, convert m1 and m2 parameters into mc and eta
        n_kl_samp=int(r*r),                        # number of iterations in statistic tests TODO: remove this
        do_adkskl_test=True,                  # if True, do statistic tests
        do_m1_m2_cut=False,                   # if True, make a cut on all m1 and m2 values    
        do_extra_noise=True,                  # add extra noise realizations during training
        Npp = int(r*r),                             # number of test signals per pp-plot. TODO: use same 
                                              # use same samples as bilby
        samplers=['vitamin','dynesty','emcee','ptemcee','cpnest'],          # list of available bilby samplers to use
        use_samplers = [0,1,2,3,4],                  # number of Bilby samplers to use 
        do_only_test = True,                  # if true, don't train but only run on test samples using pretrained network
        load_plot_data = True,                # if true, load in previously generated plot data, otherwise generate plots from scratch
        whitening_factor = np.sqrt(float(ndata)) # whitening scale factor
    )
    return params

params=get_params()

# load in y normscale
hf = h5py.File('plotting_data_%s/y_normscale_value.h5' % params['run_label'], 'r')
y_normscale = np.array(hf['y_normscale'])
hf.close()

# Make directory for plots
plots.make_dirs(params,params['plot_dir'][0])

# Declare plot class variables
plotter = plots.make_plots(params,None,None,None)

# Make KL plot
plotter.gen_kl_plots(VICI_inverse_model,None,None,None)

# Make pp plot
plotter.plot_pp(VICI_inverse_model,None,None,0,None,None,None)

# Make corner plots
plotter.make_corner_plot(None,None,sampler='dynesty1')
