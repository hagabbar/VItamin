from __future__ import division
from decimal import *
import os, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np
from scipy.stats import uniform, norm, gaussian_kde, ks_2samp, anderson_ksamp
from scipy import stats
import scipy
from scipy.integrate import dblquad
import h5py
from ligo.skymap.plot import PPPlot
import bilby
from universal_divergence import estimate
import pandas as pd
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, FixedLocator,
                               AutoMinorLocator)
import matplotlib.ticker as ticker

from data import chris_data as data_maker
#from VICI_code_usage_example import prune_samples
#from Models import VICI_inverse_model
#from Models import CVAE

"""
def prune_samples(chain_file_loc,params):
    nsteps = 5000
    nburnin = 2000
    nwalkers = 250
    thresh_num = 50
    ndim=len(params['inf_pars'])
    chain_file = pd.read_csv(chain_file_loc, sep="\t", dtype=np.float64)
#    chain_file = h5py.File(chain_file_loc, 'r')

    # Iterate over all parameters in chain file
    XS = np.array([])
    chain_file_header = list(chain_file.columns.values)
    for par_idx,par in enumerate(params['inf_pars']):
#        print(list(chain_file.columns.values))
#        print(chain_file.values.shape)
#        chains_before = np.array(chain_file[params['inf_pars'][idx]+'_post']).reshape((nsteps-nburnin,nwalkers))
#        logL = np.array(chain_file['log_like_eval']).reshape((nsteps-nburnin,nwalkers))
        chains_before = np.array(chain_file.loc[:,par])#.reshape((nsteps,nwalkers))

        if par_idx == 0:
            XS = np.expand_dims(chains_before[-(nsteps-nburnin)*nwalkers:],0)
        else:
            XS = np.vstack((XS,np.expand_dims(chains_before[-(nsteps-nburnin)*nwalkers:],0)))
        #XS = np.append(XS,np.expand_dims(chains_before,0))

    logL = np.array(chain_file.values[:,-2]).reshape((nsteps,nwalkers))
    logL = logL[-(nsteps-nburnin):,:]
    logL_max = np.max(logL)

    # data starts as (nsteps*nwalkers) x ndim -> 2D
#    XS = XS.transpose()                                     # now ndim x (nsteps*nwalkers) -> 2D
    XS = XS.reshape(ndim,nwalkers,nsteps-nburnin)                      # now ndim x nwalkers x nsteps -> 3D
    XSex = XS[:,0,:].squeeze().transpose()        # take one walker nsteps x ndim -> 2D
    XS = XS.transpose((2,1,0))                          # now nsteps x nwalkers x ndim -> 3D

    # identify good walkers
    # logL starts off with shape (nsteps*nwalkers) -> 1D
    thresh = logL_max - thresh_num                                 # define log likelihood threshold
    idx_walkers = np.argwhere([np.all(logL[:,i]>thresh) for i in range(nwalkers)])       # get the indices of good chains
    Nsamp = len(idx_walkers)*(nsteps-nburnin)                                 # redefine total number of good samples 

    # select good walkers
    XS = np.array([XS[:,i,:] for i in idx_walkers]).squeeze()     # just pick out good walkers

    XS = XS.reshape(-1,ndim)                                    # now back to original shape (but different order) (walkers*nstep) x 
    idx = np.random.choice(Nsamp,10000)          # choose 10000 random indices for corner plots

        # pick out random samples from clean set
    XS = XS[idx,:]                                                  # select 10000 random samples

    return XS
"""

def prune_samples(chain_file_loc,params):
    """ Function to remove bad likelihood emcee chains 
    """
    nsteps = 14000
    nburnin = 4000
    nwalkers = 250
    thresh_num = 50
    ndim=len(params['inf_pars'])
    chain_file = h5py.File(chain_file_loc, 'r')

    # Iterate over all parameters in chain file
    XS = np.array([])
    for idx in range(ndim):
#        print(chain_file)
#        print(params['inf_pars'][idx]+'_post')
        chains_before = np.array(chain_file[params['inf_pars'][idx]+'_post']).reshape((nsteps-nburnin,nwalkers))
        logL = np.array(chain_file['log_like_eval']).reshape((nsteps-nburnin,nwalkers))
        logL_max = np.max(logL)

        XS = np.append(XS,np.expand_dims(chains_before,0))

    # data starts as (nsteps*nwalkers) x ndim -> 2D
    XS = XS.transpose()                                     # now ndim x (nsteps*nwalkers) -> 2D
    XS = XS.reshape(ndim,nwalkers,nsteps-nburnin)                      # now ndim x nwalkers x nsteps -> 3D
    XSex = XS[:,0,:].squeeze().transpose()        # take one walker nsteps x ndim -> 2D
    XS = XS.transpose((2,1,0))                          # now nsteps x nwalkers x ndim -> 3D

    # identify good walkers
    # logL starts off with shape (nsteps*nwalkers) -> 1D
    thresh = logL_max - thresh_num                                # define log likelihood threshold
    idx_walkers = np.argwhere([np.all(logL[:,i]>thresh) for i in range(nwalkers)])       # get the indices of good chains
    Nsamp = len(idx_walkers)*(nsteps-nburnin)                                 # redefine total number of good samples 

    # select good walkers
    XS = np.array([XS[:,i,:] for i in idx_walkers]).squeeze()     # just pick out good walkers

    XS = XS.reshape(-1,ndim)                                    # now back to original shape (but different order) (walkers*nstep) x 
    idx = np.random.choice(Nsamp,10000)          # choose 10000 random indices for corner plots

        # pick out random samples from clean set
    XS = XS[idx,:]                                                  # select 10000 random samples

    return XS

def make_dirs(params,out_dir):
    """
    Make directories to store plots. Directories that already exist will be overwritten.
    """

    ## If file exists, delete it ##
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    else:    ## Show a message ##
        print("Attention: %s directory not found, making new directory" % out_dir)

    ## If file exists, delete it ##
    if os.path.exists('plotting_data_%s' % params['run_label']):
        print('plotting data directory already exits.')
    else:    ## Show a message ##
        print("Attention: plotting_data_%s directory not found, making new directory ..." % params['run_label'])
        # setup output directory - if it does not exist
        os.makedirs('plotting_data_%s' % params['run_label'])
        print('Created directory: plotting_data_%s' % params['run_label'])

    os.makedirs('%s' % out_dir)
    os.makedirs('%s/latest' % out_dir)
    os.makedirs('%s/animations' % out_dir)
   
    print('Created directory: %s' % out_dir)
    print('Created directory: %s' % (out_dir+'/latest'))
    print('Created directory: %s' % (out_dir+'/animations'))

    return


class make_plots:
    """
    Generate plots
    """
    
    def __init__(self,params,samples,rev_x,pos_test):
        """
        Add variables here later if need be
        """
        self.params = params
        self.samples = samples
        self.rev_x = rev_x
        self.pos_test = pos_test

        def ad_ks_test(parnames,inn_samps,mcmc_samps,cnt):
            """
            Record and print ks and AD test statistics
            """
    
            ks_mcmc_arr = []
            ks_inn_arr = []
            ad_mcmc_arr = []
            ad_inn_arr = []
            cur_max = self.params['n_samples']
            mcmc = []
            c=vici = []
            for i in range(inn_samps.shape[0]):
                # remove samples outside of the prior mass distribution
                mask = [(inn_samps[0,:] >= inn_samps[2,:]) & (inn_samps[3,:] >= 0.0) & (inn_samps[3,:] <= 1.0) & (inn_samps[1,:] >= 0.0) & (inn_samps[1,:] <= 1.0) & (inn_samps[0,:] >= 0.0) & (inn_samps[0,:] <= 1.0) & (inn_samps[2,:] <= 1.0) & (inn_samps[2,:] >= 0.0)]
                mask = np.argwhere(mask[0])
                new_rev = inn_samps[i,mask]
                new_rev = new_rev.reshape(new_rev.shape[0])
                new_samples = mcmc_samps[mask,i]
                new_samples = new_samples.reshape(new_samples.shape[0])
                tmp_max = new_rev.shape[0]
                if tmp_max < cur_max: cur_max = tmp_max
                vici.append(new_rev[:cur_max])
                mcmc.append(new_samples[:cur_max])

            mcmc = np.array(mcmc)
            vici = np.array(vici)

            # iterate through each parameter
            for i in range(inn_samps.shape[0]):
                ks_mcmc_samps = []
                ks_inn_samps = []
                ad_mcmc_samps = []
                ad_inn_samps = []
                n_samps = self.params['n_samples']
                n_pars = self.params['ndim_x']

                # iterate over number of randomized sample slices
                for j in range(self.params['n_kl_samp']):
                    # get ideal bayesian number. We want the 2 tailed p value from the KS test FYI
                    ks_mcmc_result = ks_2samp(np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0)), np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0)))
                    ad_mcmc_result = anderson_ksamp([np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0)), np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0))])
                

                    # get predicted vs. true number
                    ks_inn_result = ks_2samp(np.random.choice(vici[i,:],size=int(mcmc.shape[1]/2.0)),np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0)))
                    ad_inn_result = anderson_ksamp([np.random.choice(vici[i,:],size=int(mcmc.shape[1]/2.0)),np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0))])

                    # store result stats
                    ks_mcmc_samps.append(ks_mcmc_result[1])
                    ks_inn_samps.append(ks_inn_result[1])
                    ad_mcmc_samps.append(ad_mcmc_result[0])
                    ad_inn_samps.append(ad_inn_result[0])
                print('Test Case %d, Parameter(%s) k-s result: [Ideal(%.6f), Predicted(%.6f)]' % (int(cnt),parnames[i],np.array(ks_mcmc_result[1]),np.array(ks_inn_result[1])))
                print('Test Case %d, Parameter(%s) A-D result: [Ideal(%.6f), Predicted(%.6f)]' % (int(cnt),parnames[i],np.array(ad_mcmc_result[0]),np.array(ad_inn_result[0])))

                # store result stats
                ks_mcmc_arr.append(ks_mcmc_samps)
                ks_inn_arr.append(ks_inn_samps)
                ad_mcmc_arr.append(ad_mcmc_samps)
                ad_inn_arr.append(ad_inn_samps)

            return ks_mcmc_arr, ks_inn_arr, ad_mcmc_arr, ad_inn_arr, 0, 0

        def load_test_set(model,sig_test,par_test,y_normscale,bounds,sampler='dynesty1',vitamin_pred_made=None):
            """
            load requested test set
            """

            if sampler=='vitamin1' or sampler=='vitamin2':

                # check if vitamin test posteriors have already been generated
                if vitamin_pred_made != None:
                    return vitamin_pred_made[0], vitamin_pred_made[1]

                VI_pred_all = []
                for i in range(params['r']*params['r']):
                    # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                    VI_pred, dt, _  = model.run(params, np.expand_dims(sig_test[i],axis=0), np.shape(par_test)[1],
                                                             y_normscale,
                                                             "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
                    VI_pred_all.append(VI_pred)

                    print('Generated vitamin preds %d/%d' % (int(i),int(params['r']*params['r'])))

                VI_pred_all = np.array(VI_pred_all)

                return VI_pred_all, dt


            # load up the posterior samples (if they exist)
            # load generated samples back in
            post_files = []

            # choose directory with lowest number of total finished posteriors
            num_finished_post = int(1e8)
            for i in self.params['samplers']:
                if i == 'vitamin':
                    continue
                for j in range(1):
                    input_dir = '%s_%s%d/' % (self.params['pe_dir'],i,j+1)
                    if type("%s" % input_dir) is str:
                        dataLocations = ["%s" % input_dir]

                    filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))      
                    if len(filenames) < num_finished_post:
                        sampler_loc = i + str(j+1)
                        num_finished_post = len(filenames)

            dataLocations_try = '%s_%s' % (self.params['pe_dir'],sampler_loc)
            
            dataLocations = '%s_%s' % (self.params['pe_dir'],sampler)

            #for i,filename in enumerate(glob.glob(dataLocations[0])):
            i_idx = 0
            i = 0
            i_idx_use = []
            dt = []
            while i_idx < self.params['r']*self.params['r']:

                filename_try = '%s/%s_%d.h5py' % (dataLocations_try,self.params['bilby_results_label'],i)
                filename = '%s/%s_%d.h5py' % (dataLocations,self.params['bilby_results_label'],i)

                # If file does not exist, skip to next file
                try:
                    h5py.File(filename_try, 'r')
                except Exception as e:
                    i+=1
                    continue

                print(filename)
                dt.append(np.array(h5py.File(filename, 'r')['runtime']))

                post_files.append(filename)
                if sampler == 'emcee1':
                    emcee_pruned_samples = prune_samples(filename,self.params)
                data_temp = {}
                n = 0
                for q_idx,q in enumerate(self.params['inf_pars']):
                     p = q + '_post'
                     par_min = q + '_min'
                     par_max = q + '_max'
                     if p == 'psi_post':
                         data_temp[p] = np.remainder(data_temp[p],np.pi)
                     if sampler == 'emcee1':
                         data_temp[p] = emcee_pruned_samples[:,q_idx]
                     else:
                         data_temp[p] = h5py.File(filename, 'r')[p][:]
                     if p == 'geocent_time_post' or p == 'geocent_time_post_with_cut':
                         data_temp[p] = data_temp[p] - self.params['ref_geocent_time']
                     data_temp[p] = (data_temp[p] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
                     Nsamp = data_temp[p].shape[0]
                     n = n + 1

                XS = np.zeros((Nsamp,n))
                j = 0
                for p,d in data_temp.items():
                    XS[:,j] = d
                    j += 1

                #rand_idx_posterior = np.random.choice(np.linspace(0,XS.shape[0]-1,dtype=np.int),self.params['n_samples'])
                #rand_idx_posterior = np.random.choice(np.linspace(0,10000,dtype=np.int),self.params['n_samples']) 
                if i_idx == 0:
                    #XS_all = np.expand_dims(XS[rand_idx_posterior,:], axis=0)
                    XS_all = np.expand_dims(XS[:self.params['n_samples'],:], axis=0)
                else:
                    # save all posteriors in array
                    max_allow_idx = np.min([XS_all.shape[1],np.expand_dims(XS[:self.params['n_samples'],:], axis=0).shape[1]])
                    #XS_all = np.vstack((XS_all[:,:max_allow_idx,:],np.expand_dims(XS[rand_idx_posterior,:], axis=0)[:,:max_allow_idx,:]))
                    XS_all = np.vstack((XS_all[:,:max_allow_idx,:],np.expand_dims(XS[:self.params['n_samples'],:], axis=0)[:,:max_allow_idx,:]))

                i_idx_use.append(i)
                i+=1
                i_idx+=1

            # save time per sample
            dt = np.array(dt)
            dt = np.array([np.min(dt),np.max(dt),np.median(dt)])

            return XS_all, dt

        def confidence_bd(samp_array):
            """
            compute confidence bounds for a given array
            """
            cf_bd_sum_lidx = 0
            cf_bd_sum_ridx = 0
            cf_bd_sum_left = 0
            cf_bd_sum_right = 0
            cf_perc = 0.05

            cf_bd_sum_lidx = np.sort(samp_array)[int(len(samp_array)*cf_perc)]
            cf_bd_sum_ridx = np.sort(samp_array)[int(len(samp_array)*(1.0-cf_perc))]

            return [cf_bd_sum_lidx, cf_bd_sum_ridx]

        def make_contour_plot(ax,x,y,dataset,parnames,prior_min=0,prior_max=1,color='red',load_plot_data=False,contours=None):
            """ Module used to make contour plots in pe scatter plots.

            Parameters
            ----------
            ax: matplotlib figure
                a matplotlib figure instance
            x: 1D numpy array
                pe sample parameters for x-axis
            y: 1D numpy array
                pe sample parameters for y-axis
            dataset: 2D numpy array
                array containing both parameter estimates
            color:
                color of contours in plot
            Returns
            -------
            kernel: scipy kernel
                gaussian kde of the input dataset
            """
              
            def get_contours(x,y,prior_min=[0,0],prior_max=[1,1],mass_flag=False):

                #idx = np.argwhere((x>=0.0)*(x<=1.0)*(y>=0.0)*(y<=1.0)).flatten()
                #x = x[idx]
                #y = y[idx]
                N = len(x)

                values = np.vstack([x,y])
                kernel = gaussian_kde(values)
                f = lambda b, a: kernel(np.vstack([a,b]))
                if mass_flag:
                    R = dblquad(f, prior_min[0], prior_max[0], lambda x: prior_min[1], lambda x: x)[0]
                    dist = lambda a, b: f(b,a)/R*(b>=prior_min[1])*(b<=prior_max[1])*(a>=prior_min[0])*(a<=prior_max[0])*(a>=b)
                else:
                    R = dblquad(f, prior_min[0], prior_max[0], lambda x: prior_min[1], lambda x: prior_max[1])[0]
                    dist = lambda a, b: f(b,a)/R*(b>=prior_min[1])*(b<=prior_max[1])*(a>=prior_min[0])*(a<=prior_max[0])
                    #R = dblquad(f, 0, 1, lambda x: 1, lambda x: 1)
                    #dist = lambda a, b: f(b,a)/R*(b>=0)*(b<=1)*(a>=0)*(a<=1)

                Z = dist(x,y)
                Lidx = np.argsort(Z)
                Z68 = Z[Lidx[int((1.0-0.68)*N)]]
                Z90 = Z[Lidx[int((1.0-0.90)*N)]]
                Z95 = Z[Lidx[int((1.0-0.95)*N)]]

                x_range = np.max(x) - np.min(x)
                y_range = np.max(y) - np.min(y)
                xv = np.arange(np.min(x)-0.1*x_range, np.max(x)+0.1*x_range, 1.2*x_range/100.0)
                yv = np.arange(np.min(y)-0.1*y_range, np.max(y)+0.1*y_range, 1.2*y_range/100.0)
                X, Y = np.meshgrid(xv, yv)
                Q = dist(X.flatten(),Y.flatten()).reshape(X.shape)

                return Q,X,Y,[Z95,Z90,Z68,np.max(Q)]

            do_unnorm_contours = False

            if do_unnorm_contours: 
                # Make a 2d normed histogram
                H,xedges,yedges=np.histogram2d(x,y,bins=20,normed=True)

                norm=H.sum() # Find the norm of the sum
                # Set contour levels
                contour1=0.95
                contour2=0.90
                contour3=0.68

                # Set target levels as percentage of norm
                target1 = norm*contour1
                target2 = norm*contour2
                target3 = norm*contour3

                # Take histogram bin membership as proportional to Likelihood
                # This is true when data comes from a Markovian process
                def objective(limit, target):
                    w = np.where(H>limit)
                    count = H[w]
                    return count.sum() - target

                # Find levels by summing histogram to objective
                level1= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
                level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
                level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))

                # For nice contour shading with seaborn, define top level
                level4=H.max()
                levels=[level1,level2,level3,level4]

                # Pass levels to normed kde plot
                X, Y = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j]
                positions = np.vstack([X.ravel(), Y.ravel()])
                kernel = gaussian_kde(dataset)
                Z = np.reshape(kernel(positions).T, X.shape)

                if color == 'blue':
                    ax.contour(X,Y,Z,levels=levels,alpha=0.5,colors=color)
                elif color == 'red':
                    ax.contourf(X,Y,Z,levels=levels,alpha=1.0,colors=['#e61a0b','#f75448','#ff7a70'])

            else:
#                if load_plot_data==False:
                if (parnames[0] == 'm_{1} (M_\odot)' and parnames[1]=='m_{2} (M_\odot)') or (parnames[0]=='m_{2} (M_\odot)' and parnames[1]=='m_{1} (M_\odot)'):
                    mass_flag=True
                else:
                    mass_flag=False
                # Get contours for plotting
                Q,X,Y,L = get_contours(x,y,prior_min=prior_min,prior_max=prior_max,mass_flag=mass_flag)
#                else:
#                    Q = contours[0]
#                    X = contours[1]
#                    Y = contours[2]
#                    L = contours[3]
                    
                if color == 'blue':
                    ax.contour(X,Y,Q,levels=L,alpha=0.5,colors=color, origin='lower', linewidths=1)
                elif color == 'red':
                    ax.contourf(X,Y,Q,levels=L,alpha=1.0,colors=['#e61a0b','#f75448','#ff7a70'], origin='lower')
                ax.set_xlim(np.min(X),np.max(X))
                ax.set_ylim(np.min(Y),np.max(Y))
            return [Q,X,Y,L]

        # Store above declared functions to be used later
        self.ad_ks_test = ad_ks_test
        self.load_test_set = load_test_set
        self.confidence_bd = confidence_bd
        self.make_contour_plot = make_contour_plot

    def plot_testdata(self,y,s,N,outdir):
        """
        Plot the test data timeseries

        y:
            noisy time series
        s:
            noise free time series
        N:
            total number of samples?
        outdir:
            output directory
        """
        cnt = 0
        r1 = int(np.sqrt(N))
        r2 = int(N/r1)
        print(r1,r2)
        fig, axes = plt.subplots(r1,r2,figsize=(6,6),sharex='col',sharey='row',squeeze=False)
        for i in range(r1):
            for j in range(r2):
                axes[i,j].plot(y[cnt,:],'-k')
                axes[i,j].plot(s[cnt,:],'-r')
                #axes[i,j].set_xlim([0,1])
                axes[i,j].set_xlabel('t') if i==r1-1 else axes[i,j].set_xlabel('')
                axes[i,j].set_ylabel('y') if j==0 else axes[i,j].set_ylabel('')
                cnt += 1
        plt.savefig('%s/latest/test_data.png' % outdir, dpi=360)
        plt.close()

        return

    def pp_plot(self,truth,samples):
        """
        generates the pp plot data given samples and truth values
        """
        Nsamp = samples.shape[0]

        #kernel = gaussian_kde(samples.transpose())
        #v = kernel.pdf(truth)
        #x = kernel.pdf(samples.transpose())
        #r = np.sum(x>v)/float(Nsamp)
        r = np.sum(samples>truth)/float(Nsamp)

        return r

    def plot_bilby_pp(self,model,sig_test,par_test,i_epoch,normscales,pos_test,bounds):
        """ Function to make a pp plot using built-in bilby functionality
        """
        from bilby.core.prior import Uniform

        Npp = int(self.params['r']*self.params['r']) # number of test GW waveforms to use to calculate PP plot
        ndim_y = self.params['ndata']
#        priors = {f"x{jj}": Uniform(0, 1, f"x{jj}") for jj in range(len(self.params['inf_pars'])*len(self.params['samplers']))}       
        priors = {f"x{jj}": Uniform(0, 1, f"x{jj}") for jj in range(len(self.params['inf_pars']))}
        samplers = self.params['samplers']
        CB_color_cycle=['r-','b-','g-','c-','m-']
        lines = np.array([np.repeat(CB_color_cycle[jj],len(self.params['inf_pars'])) for jj in range(len(CB_color_cycle))]).flatten()

        results = []

        """
        sampler_posteriors = {}
        # Get every other sampler results
        for i in range(len(self.params['samplers'])):
            if samplers[i] == 'vitamin': continue

            # load bilby sampler samples
            samples,time = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=samplers[i]+'1') 
            if samples.shape[0] == self.params['r']**2:
                samples = samples[:,:,-self.params['n_samples']:]
            else:
                samples = samples[:self.params['n_samples'],:]
            #sampler_posteriors[samplers[i]] = samples
        """       

        # All in one plot approach
        for cnt in range(Npp):
            posterior = dict()
            injections = dict()

            #x = samples[cnt].T

            
            # Get Vitamin results
            y = sig_test[cnt,:].reshape(1,sig_test.shape[1],sig_test.shape[2])

            x, dt, _  = model.run(self.params, y, np.shape(par_test)[1],
                                                     normscales,
                                                     "inverse_model_dir_%s/inverse_model.ckpt" % self.params['run_label'])       
            # Apply mask
            x = x.T
            sampset_1 = x
            del_cnt = 0
            # iterate over each sample   during inference training
            for i in range(sampset_1.shape[1]):
                # iterate over each parameter
                for k,q in enumerate(self.params['inf_pars']):
                    # if sample out of range, delete the sample                                              the y data (size changes by factor of  n_filter/(2**n_redsteps) )
                    if sampset_1[k,i] < 0.0 or sampset_1[k,i] > 1.0:
                        x = np.delete(x,del_cnt,axis=1)
                        del_cnt-=1
                        break
                    # check m1 > m2
                    elif q == 'mass_1' or q == 'mass_2':
                        m1_idx = np.argwhere(self.params['inf_pars']=='mass_1')
                        m2_idx = np.argwhere(self.params['inf_pars']=='mass_2')
                        if sampset_1[m1_idx,i] < sampset_1[m2_idx,i]:
                            x = np.delete(x,del_cnt,axis=1)
                            del_cnt-=1
                            break
                del_cnt+=1
            

            """
            # Get every other sampler results
            for i in range(len(self.params['samplers'])):
                if samplers[i] == 'vitamin': continue

                samples = sampler_posteriors[samplers[i]]

                x = np.vstack((x,np.transpose(samples[cnt])))
            """

            dummy_idx = 0
            #scalar_truths = np.array([pos_test[cnt] for jj in range(len(self.params['samplers']))]).flatten()
            scalar_truths = pos_test[cnt]
            for key, prior in priors.items():
                posterior[key] = x[dummy_idx,:]
                injections[key] = scalar_truths[dummy_idx]
                dummy_idx+=1

            posterior = pd.DataFrame(dict(posterior))
            result = bilby.result.Result(
                label="test",
                injection_parameters=injections,
                posterior=posterior,
                search_parameter_keys=injections.keys(),
                priors=priors)
            results.append(result)

            print()
            print("Calculated vitamin pp result %d" % cnt)
            print()

        fig, pvals_bilby = bilby.result.make_pp_plot(results, filename=f"/home/hunter.gabbard/public_html/bilby_generated_pp.png",
                              confidence_interval=0.9,title=False,color='red',legend_fontsize=6.0)
        print()
        print('Made bilby pp plot')
        print()

        return

    def plot_pp(self,model,sig_test,par_test,i_epoch,normscales,pos_test,bounds):
        """
        make p-p plots using in-house methods
        
        ##########
        Parameters
        ##########
        pos_test:
            True scalar values for GW test parameters


        """
        matplotlib.rc('text', usetex=True)
        Npp = int(self.params['r']*self.params['r']) # number of test GW waveforms to use to calculate PP plot
        ndim_y = self.params['ndata']
        
        fig, axis = plt.subplots(1,1,figsize=(6,6))

        if self.params['load_plot_data'] == True:
            # Create dataset to save PP results for later plotting
            hf = h5py.File('plotting_data_%s/pp_plot_data.h5' % self.params['run_label'], 'r')
        else:
            # Create dataset to save PP results for later plotting
            try:
                os.remove('plotting_data_%s/pp_plot_data.h5' % self.params['run_label'])
            except:
                pass
            hf = h5py.File('plotting_data_%s/pp_plot_data.h5' % self.params['run_label'], 'w')

        if self.params['load_plot_data'] == False:
            pp = np.zeros(((self.params['r']**2)+2,len(self.params['inf_pars']))) 
            for cnt in range(Npp):

                # generate Vitamin samples
                if self.params['reduce'] == True or self.params['n_filters_r1'] != None:
                    y = sig_test[cnt,:].reshape(1,sig_test.shape[1],sig_test.shape[2])
                else:
                    y = sig_test[cnt,:].reshape(1,sig_test.shape[1])
                 # The trained inverse model weights can then be used to infer a probability density of solutions 
#given new measurements
                x, dt, _  = model.run(self.params, y, np.shape(par_test)[1],
                                                     normscales,
                                                     "inverse_model_dir_%s/inverse_model.ckpt" % self.params['run_label'])

                # Apply mask
                x = x.T
                sampset_1 = x   
                del_cnt = 0
                # iterate over each sample   during inference training
                for i in range(sampset_1.shape[1]):
                    # iterate over each parameter
                    for k,q in enumerate(self.params['inf_pars']):
                        # if sample out of range, delete the sample                                              the y data (size changes by factor of  n_filter/(2**n_redsteps) )
                        if sampset_1[k,i] < 0.0 or sampset_1[k,i] > 1.0:
                            x = np.delete(x,del_cnt,axis=1)   
                            del_cnt-=1
                            break
                        # check m1 > m2
                        elif q == 'mass_1' or q == 'mass_2':
                            m1_idx = np.argwhere(self.params['inf_pars']=='mass_1')
                            m2_idx = np.argwhere(self.params['inf_pars']=='mass_2')
                            if sampset_1[m1_idx,i] < sampset_1[m2_idx,i]:
                                x = np.delete(x,del_cnt,axis=1)
                                del_cnt-=1
                                break    
                    del_cnt+=1

                for j in range(len(self.params['inf_pars'])):
                    pp[0,j] = 0.0
                    pp[1,j] = 1.0
                    pp[cnt+2,j] = self.pp_plot(pos_test[cnt,j],x[j,:])
#                        pp[cnt+2] = self.pp_plot(pos_test[cnt,j],x[j,int(cnt*self.params['n_samples']):int((cnt+1)*self.params['n_samples'])])
                    print('Computed param %d p-p plot iteration %d/%d' % (j,int(cnt)+1,int(Npp)))

            # Save VItamin pp curves
            hf.create_dataset('vitamin_pp_data', data=pp)

        else:
            pp = hf['vitamin_pp_data']
            print('Loaded VItamin pp curves')

        
        confidence_pp = np.zeros((len(self.params['samplers'])-1,int(self.params['r']**2)+2))
        # plot the pp plot
        for j in range(len(self.params['inf_pars'])):        
            if j == 0:
                axis.plot(np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0),np.sort(pp[:,j]),'-',color='red',linewidth=1,zorder=50,label=r'$\textrm{%s}$' % self.params['figure_sampler_names'][0],alpha=0.5)
            else:
                axis.plot(np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0),np.sort(pp[:,j]),'-',color='red',linewidth=1,zorder=50,alpha=0.5)

        # make bilby p-p plots
        samplers = self.params['samplers']
        CB_color_cycle=['blue','green','purple','orange']

        for i in range(len(self.params['samplers'])):
            if samplers[i] == 'vitamin': continue

            if self.params['load_plot_data'] == False:
                # load bilby sampler samples
                samples,time = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=samplers[i]+'1')
                if samples.shape[0] == self.params['r']**2:
                    samples = samples[:,:,-self.params['n_samples']:]
                else:
                    samples = samples[:self.params['n_samples'],:]
                #samples = samples.reshape(self.params['r']**2,len(self.params['inf_pars']),samples.shape[1])

            for j in range(len(self.params['inf_pars'])):
                pp_bilby = np.zeros((self.params['r']**2)+2)
                pp_bilby[0] = 0.0
                pp_bilby[1] = 1.0
                if self.params['load_plot_data'] == False:
                    for cnt in range(self.params['r']**2):
                        pp_bilby[cnt+2] = self.pp_plot(pos_test[cnt,j],samples[cnt,:,j].transpose())
                        print('Computed %s, param %d p-p plot iteration %d/%d' % (samplers[i],j,int(cnt)+1,int(self.params['r']**2)))
                    hf.create_dataset('%s_param%d_pp' % (samplers[i],j), data=pp_bilby)           
                else:
                    pp_bilby = hf['%s_param%d_pp' % (samplers[i],j)]
                    print('Loaded Bilby sampler pp curve')
                
                # plot bilby sampler results
                if j == 0:
                    axis.plot(np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0),np.sort(pp_bilby),'-',color=CB_color_cycle[i-1],linewidth=1,label=r'$\textrm{%s}$' % self.params['figure_sampler_names'][i],alpha=0.5)
                else:
                    axis.plot(np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0),np.sort(pp_bilby),'-',color=CB_color_cycle[i-1],linewidth=1,alpha=0.5)

            confidence_pp[i-1,:] = np.sort(pp_bilby)

        matplotlib.rc('text', usetex=True) 
        # Remove whitespace on x-axis in all plots
        axis.margins(x=0,y=0)

        axis.plot([0,1],[0,1],'--k')
        conf_color_wheel = ['#D8D8D8','#A4A4A4','#6E6E6E']
        confidence = [0.9,0.5]
        #x_values = np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0)
        x_values = np.linspace(0, 1, 1001)
        N = int(self.params['r']**2)

        """
        # Add credibility interals
        for ci,j in zip(confidence,range(len(confidence))):
            edge_of_bound = (1. - ci) / 2.
            lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
            upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
            # The binomial point percent function doesn't always return 0 @ 0,
            # so set those bounds explicitly to be sure
            lower[0] = 0
            upper[0] = 0
            axis.fill_between(x_values, lower, upper, facecolor=conf_color_wheel[j],alpha=0.5)
        """
        axis.set_xlim([0,1])
        axis.set_ylim([0,1])
        #axis.set_ylabel(r'$\textrm{Empirical Cumulative Distribution}$',fontsize=14)
        #axis.set_xlabel(r'$\textrm{Theoretical Cumulative Distribution}$',fontsize=14)
        axis.set_ylabel(r'$\textrm{Fraction of events within the Credible Interval}$',fontsize=14)
        axis.set_xlabel(r'$\textrm{Probability within the Credible Interval}$',fontsize=14)
        axis.tick_params(axis="x", labelsize=14)
        axis.tick_params(axis="y", labelsize=14)
        #plt.axis('scaled')
        leg = axis.legend(loc='lower right', fontsize=14)
        for l in leg.legendHandles:
            l.set_alpha(1.0)
        plt.tight_layout()
        fig.savefig('%s/pp_plot_%04d.png' % (self.params['plot_dir'],i_epoch),dpi=360)
        fig.savefig('%s/latest_%s/latest_pp_plot.png' % (self.params['plot_dir'],self.params['run_label']),dpi=360)
        plt.close(fig)
        # TODO add this back in
        hf.close()
        return

    def plot_loss(self):
        """ Regenerate previously made loss plot
        """
        matplotlib.rc('text', usetex=True)

        # Load old plot data
        plotdata = np.loadtxt("inverse_model_dir_%s/loss_data.txt" % self.params['run_label'])

        # Make loss plot
        plt.figure()
        xvec = self.params['report_interval']*np.arange(np.array(plotdata).shape[0])
        plt.semilogx(xvec,np.array(plotdata)[:,0],label=r'$\mathrm{Recon}(L)$',color='blue',alpha=0.5)
        plt.semilogx(xvec,np.array(plotdata)[:,1],label=r'$\mathrm{KL}$',color='orange',alpha=0.5)
        plt.semilogx(xvec,np.array(plotdata)[:,2],label=r'$\mathrm{Total}(H)$',color='green',alpha=0.5)
        plt.semilogx(xvec,np.array(plotdata)[:,3],color='blue',linestyle='dotted')
        plt.semilogx(xvec,np.array(plotdata)[:,4],color='orange',linestyle='dotted')
        plt.semilogx(xvec,np.array(plotdata)[:,5],color='green',linestyle='dotted')
        plt.xlim([3e3,np.max(xvec)])
        plt.ylim([-25,15])
        plt.xlabel(r'$\mathrm{Iteration}$')
        plt.ylabel(r'$\mathrm{Cost}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/latest_%s/cost_%s.png' % (self.params['plot_dir'],self.params['run_label'],self.params['run_label']),dpi=360)
        plt.ylim([np.min(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,0]), np.max(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,1])])
        plt.savefig('%s/latest_%s/cost_zoom_%s.png' % (self.params['plot_dir'],self.params['run_label'],self.params['run_label']),dpi=360)
        plt.close('all')


        return

    def gen_kl_plots(self,model,sig_test,par_test,normscales,bounds,snrs_test):


        """
        Make kl corner histogram plots. Currently writing such that we 
        still bootstrap a split between samplers with themselves, but 
        will rewrite that once I find a way to run condor on 
        Bilby sampler runs.
        """
        matplotlib.rc('text', usetex=True)
        def compute_kl(sampset_1,sampset_2,samplers,one_D=False):
            """
            Compute KL for one test case.
            """
            
            # Remove samples outside of the prior mass distribution           
            cur_max = self.params['n_samples']
            
            # Iterate over parameters and remove samples outside of prior
            if samplers[0] == 'vitamin1' or samplers[1] == 'vitamin2':

                # Apply mask
                sampset_1 = sampset_1.T
                sampset_2 = sampset_2.T
                set1 = sampset_1
                set2 = sampset_2
                del_cnt_set1 = 0
                del_cnt_set2 = 0
                params_to_infer = self.params['inf_pars']
                for i in range(set1.shape[1]):

                    # iterate over each parameter in first set
                    for k,q in enumerate(params_to_infer):
                        # if sample out of range, delete the sample
                        if set1[k,i] < 0.0 or set1[k,i] > 1.0:
                            sampset_1 = np.delete(sampset_1,del_cnt_set1,axis=1)
                            del_cnt_set1-=1
                            break
                        # check m1 > m2
                        elif q == 'mass_1' or q == 'mass_2':
                            m1_idx = np.argwhere(params_to_infer=='mass_1')
                            m2_idx = np.argwhere(params_to_infer=='mass_2')
                            if set1[m1_idx,i] < set1[m2_idx,i]:
                                sampset_1 = np.delete(sampset_1,del_cnt_set1,axis=1)
                                del_cnt_set1-=1
                                break

                    del_cnt_set1+=1

                # iterate over each sample
                for i in range(set2.shape[1]):

                    # iterate over each parameter in second set
                    for k,q in enumerate(params_to_infer):
                        # if sample out of range, delete the sample
                        if set2[k,i] < 0.0 or set2[k,i] > 1.0:
                            sampset_2 = np.delete(sampset_2,del_cnt_set2,axis=1)
                            del_cnt_set2-=1
                            break
                        # check m1 > m2
                        elif q == 'mass_1' or q == 'mass_2':
                            m1_idx = np.argwhere(params_to_infer=='mass_1')
                            m2_idx = np.argwhere(params_to_infer=='mass_2')
                            if set2[m1_idx,i] < set2[m2_idx,i]:
                                sampset_2 = np.delete(sampset_2,del_cnt_set2,axis=1)
                                del_cnt_set2-=1
                                break

                    del_cnt_set2+=1

                del_final_idx = np.min([del_cnt_set1,del_cnt_set2])
                set1 = sampset_1[:,:del_final_idx]
                set2 = sampset_2[:,:del_final_idx]

            else:

                set1 = sampset_1.T
                set2 = sampset_2.T
      
            
#            plt.hist(sampset_1[:,0])
#            plt.savefig('/home/hunter.gabbard/public_html/test1.png')
#            plt.close()
#            set1 = sampset_1.T
#            set2 = sampset_2.T
#            plt.hist(set1[1,:])
#            plt.savefig('/home/hunter.gabbard/public_html/test2.png')
#            plt.close()

#            idx_set = [5,6]
#            set1 = set1[idx_set,:]
#            set2 = set2[idx_set,:]
#            kl_samps = []
#            n_samps = self.params['n_samples']
#            n_pars = len(self.params['inf_pars'])
           
            """
            set1 = set1.T
            set2 = set2.T
            def MMD_multiscale(x, y):
                xx, yy, zz = np.matmul(x,x.T), np.matmul(y,y.T), np.matmul(x,y.T)

                rx = (np.expand_dims(np.diag(xx),axis=0))
                ry = (np.expand_dims(np.diag(yy),axis=0))
                rx = np.tile(rx,(rx.shape[1],1))
                ry = np.tile(ry,(ry.shape[1],1))
                

                dxx = rx.T + rx - 2.*xx
                dyy = ry.T + ry - 2.*yy
                dxy = rx.T + ry - 2.*zz

                XX, YY, XY = (np.zeros(xx.shape),
                  np.zeros(xx.shape),
                  np.zeros(xx.shape))

                alphas = [0.05,0.2,0.5,0.9,1.3]
                for a in alphas:
                    XX += a**2 * (a**2 + dxx)**-1
                    YY += a**2 * (a**2 + dyy)**-1
                    XY += a**2 * (a**2 + dxy)**-1

                return np.mean(XX + YY - 2.*XY)

            mmd_result = MMD_multiscale(set1,set2)
            return mmd_result * 1e3

            exit()
            """
            # Iterate over number of randomized sample slices
            SMALL_CONSTANT = 1e-162 # 1e-4 works best for some reason
            def my_kde_bandwidth(obj, fac=1.0):

                """We use Scott's Rule, multiplied by a constant factor."""

                return np.power(obj.n, -1./(obj.d+4)) * fac
            if one_D:
                kl_result_all = np.zeros((1,len(self.params['inf_pars'])))
                for r in range(len(self.params['inf_pars'])):
                    if self.params['gen_indi_KLs'] == True:
                        p = gaussian_kde(set1[r],bw_method=my_kde_bandwidth)#'scott') # 7.5e0 works best ... don't know why. Hope it's not over-smoothing results.
                        q = gaussian_kde(set2[r],bw_method=my_kde_bandwidth)#'scott')#'silverman') # 7.5e0 works best ... don't know why.   
                        # Compute KL Divergence
                        log_diff = np.log((p(set1[r])+SMALL_CONSTANT)/(q(set1[r])+SMALL_CONSTANT))
                        kl_result = (1.0/float(set1.shape[1])) * np.sum(log_diff)

                        # compute symetric kl
                        anti_log_diff = np.log((q(set2[r])+SMALL_CONSTANT)/(p(set2[r])+SMALL_CONSTANT))
                        anti_kl_result = (1.0/float(set1.shape[1])) * np.sum(anti_log_diff)
                        kl_result_all[:,r] = kl_result + anti_kl_result
#                        kl_result = estimate(np.expand_dims(set1[r],1),np.expand_dims(set2[r],1))
#                        kl_result_all[:,r] = kl_result 
                    else:
                        kl_result_all[:,r] = 0   

                return kl_result_all
            else:
                
                kl_result = []
                set1 = set1.T
                set2 = set2.T
                for kl_idx in range(10):
                    rand_idx_kl = np.random.choice(np.linspace(0,set1.shape[0]-1,dtype=np.int),size=100)
                    rand_idx_kl_2 = np.random.choice(np.linspace(0,set2.shape[0]-1,dtype=np.int),size=100)
                    kl_result.append(estimate(set1[rand_idx_kl,:],set2[rand_idx_kl_2,:]) + estimate(set2[rand_idx_kl_2,:],set1[rand_idx_kl,:]))
                kl_result = np.mean(kl_result)
                return kl_result

                """
                # TODO: comment this out when not doing dynesty kl with itself. Use above expression instead.
                kl_result_mean = []
                kl_result_std = []
                set1 = set1.T
                set2 = set2.T
                for kl_idx in range(10):
                    rand_idx_kl = np.random.choice(np.linspace(0,set1.shape[0]-1,dtype=np.int),size=100)
                    rand_idx_kl_2 = np.random.choice(np.linspace(0,set2.shape[0]-1,dtype=np.int),size=100)
                    new_kl = estimate(set1[rand_idx_kl,:],set2[rand_idx_kl_2,:]) + estimate(set2[rand_idx_kl_2,:],set1[rand_idx_kl,:])
                    kl_result_mean.append(new_kl)
                    kl_result_std.append(new_kl**2)
                kl_result_mean = np.mean(kl_result_mean)
                kl_result_std = np.sqrt(np.mean(kl_result_std))
 
                return kl_result_std, kl_result_mean
                """   

        # Define variables 
        params = self.params
        usesamps = params['samplers']
        samplers = params['samplers']
        fig_samplers = params['figure_sampler_names']
        indi_fig_kl, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3,figsize=(6,6))  
        indi_axis_kl = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
      
        # Compute kl divergence on all test cases with preds vs. benchmark
        # Iterate over samplers
        tmp_idx=len(usesamps)
        print_cnt = 0
        runtime = {}
        CB_color_cycle = ['orange', 'purple', 'green',
                  'blue', '#a65628', '#984ea3',
                  '#e41a1c', '#dede00', 
                  '#004d40','#d81b60','#1e88e5',
                  '#ffc107','#1aff1a','#377eb8',
                  '#fefe62','#d35fb7','#dc3220']
        label_idx = 0
        vi_pred_made = None

        
        if params['load_plot_data'] == False:
            # Create dataset to save KL divergence results for later plotting
            try:
                os.mkdir('plotting_data_%s' % params['run_label']) 
            except:
                print('Plotting directory already exists')

            try:
                hf = h5py.File('plotting_data_%s/KL_plot_data.h5' % params['run_label'], 'w')
            except:
                os.remove('plotting_data_%s/KL_plot_data.h5' % params['run_label'])
                hf = h5py.File('plotting_data_%s/KL_plot_data.h5' % params['run_label'], 'w')
        else:
            # TODO: uncomment this
            #hf = h5py.File('plotting_data_%s/KL_plot_data.h5' % params['run_label'], 'r')
            hf = h5py.File('plotting_data_%s/KL_plot_data.h5' % params['run_label'], 'r')
        

            
        # 4 pannel KL approach
        fig_kl, axis_kl = plt.subplots(2,2,figsize=(8,6),sharey=True,sharex=True)
        for k in range(len(usesamps)-1):
            print_cnt = 0
            label_idx = 0
            tmp_idx = len(usesamps)
            if k <= 1:
               kl_idx_1 = 0
               kl_idx_2 = k
            elif k > 1:
               kl_idx_1 = 1
               kl_idx_2 = (k-2)

            tot_kl_grey = np.array([])
            for i in range(len(usesamps)):
                for j in range(tmp_idx):
                    # Load appropriate test sets
                    if samplers[i] == samplers[::-1][j]:
                        continue
                    else:
                        sampler1, sampler2 = samplers[i]+'1', samplers[::-1][j]+'1'

                    tot_kl = np.array(hf['%s-%s' % (sampler1,sampler2)])

                    logbins = np.logspace(-3,2.5,50)
                    logbins_indi = np.logspace(-3,3,50)

                    # plot colored hist
                    if samplers[i] == 'vitamin' and samplers[::-1][j] == samplers[1:][k]: 
#                        print(tot_kl.argsort()[-15:][::-1])
#                        print(np.sort(tot_kl)[-15:][::-1])
#                        print(tot_kl.argsort()[:15][:])
#                        print(np.sort(tot_kl)[:15][:])
                        axis_kl[kl_idx_1,kl_idx_2].hist(tot_kl,bins=logbins,alpha=0.5,histtype='stepfilled',density=True,color=CB_color_cycle[print_cnt],label=r'$\mathrm{%s \ vs. \ %s}$' % (fig_samplers[i],fig_samplers[::-1][j]),zorder=2)
                        axis_kl[kl_idx_1,kl_idx_2].hist(tot_kl,bins=logbins,histtype='step',density=True,facecolor='None',ls='-',lw=2,edgecolor=CB_color_cycle[print_cnt],zorder=10)
                    # record non-colored hists
                    elif samplers[i] != 'vitamin' and samplers[::-1][j] != 'vitamin':
                        if samplers[i] == samplers[1:][k] or samplers[::-1][j] == samplers[1:][k]:
#                            print(tot_kl.argsort()[-15:][::-1])
#                            print(np.sort(tot_kl)[-15:][::-1])
#                            print(tot_kl.argsort()[:15][:])
#                            print(np.sort(tot_kl)[:15][:])

                            tot_kl_grey = np.append(tot_kl_grey,tot_kl)

                            print('Mean total KL between bilby samps: %s' % str(np.mean(tot_kl)))
                    print('Completed KL calculation %d/%d' % (print_cnt,len(usesamps)*2))
                    print_cnt+=1
                tmp_idx-=1

            # Plot non-colored histograms
            axis_kl[kl_idx_1,kl_idx_2].hist(np.array(tot_kl_grey).squeeze(),bins=logbins,alpha=0.8,histtype='stepfilled',density=True,color='grey',label=r'$\mathrm{%s \ vs. \ other \ samplers}$' % fig_samplers[1:][k],zorder=1)
            axis_kl[kl_idx_1,kl_idx_2].hist(np.array(tot_kl_grey).squeeze(),bins=logbins,histtype='step',density=True,facecolor='None',ls='-',lw=2,edgecolor='grey',zorder=1)

            # plot KL histograms
            if kl_idx_1 == 1:
                axis_kl[kl_idx_1,kl_idx_2].set_xlabel(r'$\mathrm{KL-Statistic}$',fontsize=14)
            if kl_idx_2 == 0:
                axis_kl[kl_idx_1,kl_idx_2].set_ylabel(r'$p(\mathrm{KL})$',fontsize=14)
           # axis_kl[kl_idx_1,kl_idx_2].tick_params(axis="both", labelsize=12, direction='out')
            leg = axis_kl[kl_idx_1,kl_idx_2].legend(loc='upper left',  fontsize=6) #'medium')
            for l in leg.legendHandles:
                l.set_alpha(1.0)

            #axis_kl[kl_idx_1,kl_idx_2].xaxis.set_minor_locator(FixedLocator([0.5, 1.5, 2.5, 3.5, 4.5]))
            #axis_kl[kl_idx_1,kl_idx_2].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
            axis_kl[kl_idx_1,kl_idx_2].set_xlim(left=1e-2,right=100)
            ##axis_kl[kl_idx_1,kl_idx_2].set_xticks(AutoMinorLocator(),minor=True)
            #caxis_kl[kl_idx_1,kl_idx_2].xaxis.set_minor_locator(MultipleLocator(5))
            #axis_kl[kl_idx_1,kl_idx_2].tick_params(which='minor', length=4, color='r')
            #axis_kl[kl_idx_1,kl_idx_2].set_ylim(top=1.0)
            axis_kl[kl_idx_1,kl_idx_2].set_xscale('log')
            locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=25)
            axis_kl[kl_idx_1,kl_idx_2].xaxis.set_minor_locator(locmin)
            locmaj = matplotlib.ticker.LogLocator(base=10, numticks=25)
            axis_kl[kl_idx_1,kl_idx_2].xaxis.set_major_locator(locmaj)
            axis_kl[kl_idx_1,kl_idx_2].set_yscale('log')
            axis_kl[kl_idx_1,kl_idx_2].grid(False)
            print()
            print('Made hist plot %d' % k)
            print()

        # Save figure
        fig_kl.canvas.draw()
        #plt.minorticks_on()
        plt.tight_layout()
        fig_kl.savefig('%s/latest_%s/hist-kl.png' % (self.params['plot_dir'],self.params['run_label']),dpi=360)
        plt.close(fig_kl)
        hf.close()
        return
        
        """
        # Compute dynesty with itself once
        sampler1 = 'dynesty1'
        sampler2 = 'dynesty1'
        logbins = np.logspace(-3,2.5,50)
        time_dict = {}

        set1,time_dict[sampler1] = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=sampler1) 
#        set2,time_dict[sampler2] = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=sampler2)
        set2 = np.copy(set1)
        tot_kl_std = []
        tot_kl_mean = []
        for r in range(self.params['r']**2):
            std_kl, mean_kl = compute_kl(set1[r],set2[r],[sampler1,sampler2])
            tot_kl_std.append(std_kl)
            tot_kl_mean.append(mean_kl)
            print(tot_kl_std[r])
            print('Completed KL for set %s-%s and test sample %s' % (sampler1,sampler2,str(r)))
        tot_kl_std = np.array(tot_kl_std)
        tot_kl_mean = np.array(tot_kl_mean)

        # linear plot
        fig_kl, axis_kl = plt.subplots(1,1,figsize=(6,6))
        axis_kl.hist(tot_kl_std,bins=50,histtype='stepfilled',density=True,color='grey')
        axis_kl.set_xlabel(r'$\mathrm{KL-Statistic}$',fontsize=14)
        axis_kl.set_ylabel(r'$p(\mathrm{KL})$',fontsize=14)
        axis_kl.tick_params(axis="x", labelsize=14)
        axis_kl.tick_params(axis="y", labelsize=14)
        leg = axis_kl.legend(loc='upper right', fontsize=10) #'medium')
        for l in leg.legendHandles:
            l.set_alpha(1.0)

        axis_kl.set_xlim(left=1e-2,right=1)
        axis_kl.grid(False)
        fig_kl.canvas.draw()
        fig_kl.savefig('/home/hunter.gabbard/public_html/CBC/VItamin/dynesty_with_itself_linear.png')
        plt.close()

        # log plot
        fig_kl, axis_kl = plt.subplots(1,1,figsize=(6,6))
        axis_kl.hist(tot_kl_std,bins=logbins,histtype='stepfilled',density=True,color='grey')
        axis_kl.set_xlabel(r'$\mathrm{KL-Statistic}$',fontsize=14)
        axis_kl.set_ylabel(r'$p(\mathrm{KL})$',fontsize=14)
        axis_kl.tick_params(axis="x", labelsize=14)
        axis_kl.tick_params(axis="y", labelsize=14)
        leg = axis_kl.legend(loc='upper right', fontsize=10) #'medium')
        for l in leg.legendHandles:
            l.set_alpha(1.0)

        axis_kl.set_xlim(left=1e-2,right=1)
        axis_kl.set_xscale('log')
        axis_kl.set_yscale('log')
        axis_kl.grid(False)
        fig_kl.canvas.draw()
        fig_kl.savefig('/home/hunter.gabbard/public_html/CBC/VItamin/dynesty_with_itself_log.png')
        plt.close()

        print()
        print('Mean std KL is below:')
        print(np.mean(tot_kl_std))
        print()
        print('Mean KL is below: ')
        print(np.mean(tot_kl_mean))
        print('Finished computing KL with itself')
        exit()
        """

        tot_kl_grey = np.array([])
        fig_kl, axis_kl = plt.subplots(1,1,figsize=(6,6))
        time_dict = {}
        # single plot KL approach 
        for i in range(len(usesamps)):
            for j in range(tmp_idx):

                # Load appropriate test sets
                if samplers[i] == samplers[::-1][j]:
                    print_cnt+=1
                    sampler1, sampler2 = samplers[i]+'1', samplers[::-1][j]+'1'

                    # currently not doing KL of approaches with themselves, so skip here
                    continue
                    if self.params['load_plot_data'] == False:
                        set1,time_dict[sampler1] = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=sampler1)
                        set2,time_dict[sampler2] = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=sampler2)
                else:
                    sampler1, sampler2 = samplers[i]+'1', samplers[::-1][j]+'1'
                   
                    if self.params['load_plot_data'] == False:
                        set1,time_dict[sampler1] = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=sampler1,vitamin_pred_made=vi_pred_made)
                        set2,time_dict[sampler2] = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=sampler2,vitamin_pred_made=vi_pred_made)

                        # check if vitamin test posteriors were generated for the first time
                        if sampler1 == 'vitamin1' and vi_pred_made == None:
                            vi_pred_made = [set1,time_dict[sampler1]]
                        elif sampler2 == 'vitamin1' and vi_pred_made == None:
                            vi_pred_made = [set2,time_dict[sampler2]]

                if self.params['load_plot_data'] == True:
                    tot_kl = np.array(hf['%s-%s' % (sampler1,sampler2)])
                else:

                    # Iterate over test cases
                    tot_kl = []  # total KL over all infered parameters
                    indi_kl = np.zeros((self.params['r']**2,len(self.params['inf_pars']))) # KL for each individual paramter

                    if self.params['make_indi_kl'] == True:
                        for r in range(self.params['r']**2):
                            indi_kl[r,:] = compute_kl(set1[r],set2[r],[sampler1,sampler2],one_D=True)
                            print('Completed KL for set %s-%s and test sample %s' % (sampler1,sampler2,str(r)))
                    for r in range(self.params['r']**2):
                        tot_kl.append(compute_kl(set1[r],set2[r],[sampler1,sampler2]))
                        print('Completed KL for set %s-%s and test sample %s' % (sampler1,sampler2,str(r)))
                    tot_kl = np.array(tot_kl)

                if self.params['load_plot_data'] == False:

                    # Save results to h5py file
                    hf.create_dataset('%s-%s' % (sampler1,sampler2), data=tot_kl)
               
                logbins = np.logspace(-3,2.5,50)
                logbins_indi = np.logspace(-3,3,50)
                #logbins = 50

                if samplers[i] == 'vitamin' or samplers[::-1][j] == 'vitamin':
                    # print 10 worst and 10 best kl
                    print(tot_kl.argsort()[-15:][::-1])
                    print(np.sort(tot_kl)[-15:][::-1])
                    print(tot_kl.argsort()[:15][:])
                    print(np.sort(tot_kl)[:15][:])

                    # plot vitamin kls
                    axis_kl.hist(tot_kl,bins=logbins,alpha=0.5,histtype='stepfilled',density=True,color=CB_color_cycle[print_cnt],label=r'$\textrm{VItamin-%s}$' % (samplers[::-1][j]),zorder=2)
                    axis_kl.hist(tot_kl,bins=logbins,histtype='step',density=True,facecolor='None',ls='-',lw=2,edgecolor=CB_color_cycle[print_cnt],zorder=10)

                    if self.params['make_indi_kl'] == True:
                        # plot indi vitamin kls
                        for u in range(len(self.params['inf_pars'])):
                            indi_axis_kl[u].hist(indi_kl[:,u],bins=logbins_indi,alpha=0.5,histtype='stepfilled',density=True,color=CB_color_cycle[print_cnt],label=r'$\textrm{VItamin vs. %s}$' % (samplers[::-1][j]),zorder=2)
                            indi_axis_kl[u].hist(indi_kl[:,u],bins=logbins_indi,histtype='step',density=True,facecolor='None',ls='-',lw=0.5,edgecolor=CB_color_cycle[print_cnt],zorder=10)    
                        print('Mean total KL vitamin vs bilby: %s' % str(np.mean(tot_kl)))
                    
                    # Return the mean KL if doing hyperparameter optimization
                    if self.params['hyperparam_optim'] == True:
                        return np.mean(tot_kl)
                else:
                    print(tot_kl.argsort()[-15:][::-1])
                    print(np.sort(tot_kl)[-15:][::-1])
                    print(tot_kl.argsort()[:15][:])
                    print(np.sort(tot_kl)[:15][:]) 

                    
                    tot_kl_grey = np.append(tot_kl_grey,tot_kl)

                    if label_idx == 0:

                        if self.params['make_indi_kl'] == True:
                            # plot indi bayesian kls
                            for u in range(len(self.params['inf_pars'])):
                                indi_axis_kl[u].hist(indi_kl[:,u],bins=logbins_indi,alpha=0.8,histtype='stepfilled',density=True,color='grey',label=r'$\textrm{other samplers}$',zorder=1)

                        label_idx += 1
                    else:
                        if self.params['make_indi_kl'] == True:
                            # plot indi bayesian kls
                            for u in range(len(self.params['inf_pars'])):
                                indi_axis_kl[u].hist(indi_kl[:,u],bins=logbins_indi,alpha=0.8,histtype='stepfilled',density=True,color='grey',zorder=1)

                    if self.params['make_indi_kl'] == True:
                        # plot indi bayesian kls
                        for u in range(len(self.params['inf_pars'])):
                            indi_axis_kl[u].hist(indi_kl[:,u],bins=logbins_indi,histtype='step',density=True,facecolor='None',ls='-',lw=0.2,edgecolor='grey',zorder=1)
                 
                    print('Mean total KL between bilby samps: %s' % str(np.mean(tot_kl)))

                print('Completed KL calculation %d/%d' % (print_cnt,len(usesamps)*2))
                print_cnt+=1

            tmp_idx -= 1
            if self.params['load_plot_data'] == False:
                runtime[sampler1] = time_dict[sampler1]

        # plot non indi bayesian kls
        axis_kl.hist(tot_kl_grey,bins=logbins,alpha=0.8,histtype='stepfilled',density=True,color='grey',label=r'$\textrm{other samplers vs. other samplers}$',zorder=1)
        # plot non indi bayesian kls
        axis_kl.hist(tot_kl_grey,bins=logbins,histtype='step',density=True,facecolor='None',ls='-',lw=2,edgecolor='grey',zorder=1)

        if self.params['load_plot_data'] == False:
#            # Print sampler runtimes
            for i in range(len(usesamps)):
                hf.create_dataset('%s_runtime' % (samplers[i]), data=np.array(runtime[samplers[i]+'1']))
                print('%s sampler runtimes: %s' % (samplers[i]+'1',str(runtime[samplers[i]+'1'])))

        

        # Save KL histogram
        axis_kl.set_xlabel(r'$\mathrm{KL-Statistic}$',fontsize=14)
        axis_kl.set_ylabel(r'$p(\mathrm{KL})$',fontsize=14)
        axis_kl.tick_params(axis="x", labelsize=14)
        axis_kl.tick_params(axis="y", labelsize=14)
        leg = axis_kl.legend(loc='upper right', fontsize=10) #'medium')
        for l in leg.legendHandles: 
            l.set_alpha(1.0)

        axis_kl.set_xlim(left=8e-2,right=100)
        axis_kl.set_xscale('log')
        axis_kl.set_yscale('log')
        axis_kl.grid(False)
        fig_kl.canvas.draw()
#        fig_kl.savefig('%s/latest_%s/hist-kl.png' % (self.params['plot_dir'],self.params['run_label']),dpi=360)
        plt.close(fig_kl)

        if self.params['make_indi_kl'] == True:
            # save indi kl histogram
            for u in range(len(self.params['inf_pars'])):
                indi_axis_kl[u].set_xlabel(r'$\mathrm{%s}$' % self.params['inf_pars'][u],fontsize=5)
                indi_axis_kl[u].tick_params(axis="x", labelsize=5)
                indi_axis_kl[u].tick_params(axis="y", labelsize=5)

                indi_axis_kl[u].set_xlim(left=1e-3)
                indi_axis_kl[u].set_xscale('log')
                indi_axis_kl[u].set_yscale('log')
                indi_axis_kl[u].grid(False)
            indi_axis_kl[7].set_visible(False)
            indi_axis_kl[8].set_visible(False)
            indi_fig_kl.canvas.draw()
            indi_fig_kl.savefig('%s/latest_%s/hist-kl_individual_par.png' % (self.params['plot_dir'],self.params['run_label']),dpi=360)
            plt.close(indi_fig_kl)

        hf.close()
        return

    def make_corner_plot(self,noisefreeY_test,noisyY_test,bounds,test_sample_idx,epoch_idx,sampler='dynesty1'):
        """
        Function to generate a corner plot for n-test GW samples. Corner plot has posteriors 
        from two samplers (usually VItamin and some other Bayesian sampler). The 4D overlap 
        for each sample is usually posted in the upper right hand corner of each plot, 
        but is set to zero when not in use.

        """
        matplotlib.rc('text', usetex=True)
        # Define variables
        params = self.params

        # make directory to store original plotting data

        if self.params['load_plot_data'] == True:
            hf = h5py.File('plotting_data_%s/corner_plot_data.h5' % params['run_label'], 'r')
            Vitamin_preds = np.array(hf['Vitamin_preds'])
            sampler_preds = np.array(hf['sampler_preds']) 
            self.pos_test = np.array(hf['pos_test'])
            noisefreeY_test = np.array(hf['noisefreeY_test'])
            noisyY_test = np.array(hf['noisyY_test'])
        else:
            Vitamin_preds = self.rev_x
            #sampler_preds,_ = self.load_test_set(None,None,None,None,sampler=sampler)
            sampler_preds = np.array(self.samples, copy=True)

            try:
                # Save data for later plotting use
                os.mkdir('plotting_data_%s' % params['run_label'])
            except Exception as e:
                print(e)
                pass
            hf = h5py.File('plotting_data_%s/corner_plot_data.h5' % params['run_label'], 'w')
            hf.create_dataset('Vitamin_preds', data=Vitamin_preds)
            hf.create_dataset('sampler_preds', data=sampler_preds)
            hf.create_dataset('pos_test', data=self.pos_test)
            if noisefreeY_test is not None and noisyY_test is not None:
                hf.create_dataset('noisefreeY_test', data=noisefreeY_test)
                hf.create_dataset('noisyY_test', data=noisyY_test)
            contour_info = hf.create_group('contour_info')

        """
        # rescale truths
        self.pos_test[:,0] = (self.pos_test[:,0] * (params['prior_max'][0] - params['prior_min'][0])) + (params['prior_min'][0])
        self.pos_test[:,1] = (self.pos_test[:,1] * (params['prior_max'][2] - params['prior_min'][2])) + (params['prior_min'][2]) - (params['ref_geocent_time']-0.5)
        self.pos_test[:,2] = (self.pos_test[:,2] * (params['prior_max'][3] - params['prior_min'][3])) + (params['prior_min'][3])
        self.pos_test[:,3] = (self.pos_test[:,3] * (params['prior_max'][4] - params['prior_min'][4])) + (params['prior_min'][4])

        pmin = np.zeros(4)
        pmax = np.zeros(4)
        pmin[0] = params['prior_min'][0]
        pmin[1] = params['prior_min'][2] - (params['ref_geocent_time']-0.5)
        pmin[2] = params['prior_min'][3]
        pmin[3] = params['prior_min'][4] 
        pmax[0] = params['prior_max'][0]
        pmax[1] = params['prior_max'][2] - (params['ref_geocent_time']-0.5)
        pmax[2] = params['prior_max'][3]
        pmax[3] = params['prior_max'][4]
        """

        # Iterate over test samples
        for r in range(1): #range(params['r']**2):
            print('Making corner plot %d ...' % test_sample_idx)

            # Declare figure object
            fig_corner, axis_corner = plt.subplots(len(params['inf_pars']),len(params['inf_pars']),figsize=(6,6))#,sharex='col')

            set1 = Vitamin_preds[r].T
            set2 = sampler_preds[r].T

            # Apply mask
            sampset_1 = set1
            sampset_2 = set2
            print(sampset_1.shape)
            del_cnt = 0
            # iterate over each sample
            for i in range(sampset_1.shape[1]):
                # iterate over each parameter
                for j,q in enumerate(params['inf_pars']):
                    # if sample out of range, delete the sample
                    if sampset_1[j,i] < 0.0 or sampset_1[j,i] > 1.0:
                        set1 = np.delete(set1,del_cnt,axis=1)
                        set2 = np.delete(set2,del_cnt,axis=1)
                        del_cnt-=1
                        break
                    # check m1 > m2
                    elif q == 'mass_1' or q == 'mass_2':
                        m1_idx = np.argwhere(params['inf_pars']=='mass_1') 
                        m2_idx = np.argwhere(params['inf_pars']=='mass_2')
                        if sampset_1[m1_idx,i] < sampset_1[m2_idx,i]:
                            set1 = np.delete(set1,del_cnt,axis=1)
                            set2 = np.delete(set2,del_cnt,axis=1)
                            del_cnt-=1
                            break
                del_cnt+=1

            left, bottom, width, height = [0.6, 0.69, 0.3, 0.19]
            ax2 = fig_corner.add_axes([left, bottom, width, height])

           
            # plot waveform in upper-right hand corner
            ax2.plot(np.linspace(0,1,params['ndata']),noisefreeY_test[:],color='cyan',zorder=50)
            ax2.plot(np.linspace(0,1,params['ndata']),noisyY_test[:],color='darkblue')
            ax2.set_xlabel(r"$\textrm{time (seconds)}$",fontsize=11)
            ax2.yaxis.set_visible(False)
            ax2.tick_params(axis="x", labelsize=11)
            ax2.tick_params(axis="y", labelsize=11)
            ax2.set_ylim([-6,6])
            ax2.grid(False)
            ax2.margins(x=0,y=0)

            # get the mins and maxs for the prior bounds being plotted
            usepar_priormax = np.zeros((len(params['inf_pars'])))
            usepar_priormin = np.zeros((len(params['inf_pars'])))
            pos_test_copy = self.pos_test
            for q_idx,q in enumerate(params['inf_pars']):
                par_min = q + '_min'
                par_max = q + '_max'
                usepar_priormax[q_idx] = bounds[par_max]
                usepar_priormin[q_idx] = bounds[par_min]

                # rescale parameters back to their physical values
#                if par_min == 'geocent_time_min':
#                    set1[q_idx,:] = (set1[q_idx,:] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]
                    #self.pos_test[r,q_idx] = (self.pos_test[r,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]
#                    continue
                    
                set1[q_idx,:] = (set1[q_idx,:] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]
                set2[q_idx,:] = (set2[q_idx,:] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]
                #pos_test_copy[r,q_idx] = (pos_test_copy[r,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]

            # Iterate over parameters
            tmp_idx=len(params['inf_pars'])
            
            # Get corner parnames to use in plotting labels
            parnames = []
            for k_idx,k in enumerate(params['rand_pars']):
                if np.isin(k, params['inf_pars']):
                    parnames.append(params['corner_parnames'][k_idx])
            

            if self.params['load_plot_data'] == False:
                testsamp_group = contour_info.create_group('testsamp%d' % r)
            for i in range(len(params['inf_pars'])):
                for j in range(tmp_idx):
#                    overlap = data_maker.overlap(set1.T,set2.T,next_cnt=True)
                    parname1 = parnames[i]
                    usepars_order = np.arange(0,len(params['inf_pars']))
                    parname2 = parnames[usepars_order[::-1][j]]

                    axis_corner[len(params['inf_pars'])-1-j,i].clear()
                    # Make histograms on diagonal
                    if (len(params['inf_pars'])-1-j) == i:
                        axis_corner[len(params['inf_pars'])-1-j,i].hist(set1[i,:],bins=20,alpha=0.5,density=True,histtype='stepfilled',label='VItamin',color='r')
                        axis_corner[len(params['inf_pars'])-1-j,i].hist(set1[i,:],bins=20,lw=1,density=True,histtype='step',label='VItamin',color='r',zorder=20)
                        axis_corner[len(params['inf_pars'])-1-j,i].hist(set2[i,:],bins=20,alpha=0.5,density=True,histtype='stepfilled',label=sampler,color='b')
                        axis_corner[len(params['inf_pars'])-1-j,i].hist(set2[i,:],bins=20,lw=1,density=True,histtype='step',label=sampler,color='b',zorder=10)
                        axis_corner[len(params['inf_pars'])-1-j,i].axvline(x=pos_test_copy[r,i], linewidth=1.0, color='black',zorder=30)
                        axis_corner[len(params['inf_pars'])-1-j,i].axvline(x=self.confidence_bd(set1[i,:])[0], linewidth=1, color='r',zorder=30)
                        axis_corner[len(params['inf_pars'])-1-j,i].axvline(x=self.confidence_bd(set1[i,:])[1], linewidth=1, color='r',zorder=30)
                        axis_corner[len(params['inf_pars'])-1-j,i].axvline(x=self.confidence_bd(set2[i,:])[0], linewidth=1, color='b',zorder=30)
                        axis_corner[len(params['inf_pars'])-1-j,i].axvline(x=self.confidence_bd(set2[i,:])[1], linewidth=1, color='b',zorder=30)

                        xmin, xmax = axis_corner[len(params['inf_pars'])-1-j,i].get_xlim()
                        ymin, ymax = axis_corner[len(params['inf_pars'])-1-j,i].get_ylim()
                        #if ymin<pmin[params['ndim_x']-1-j]:
                        #    axis_corner[params['ndim_x']-1-j,i].set_ylim(ymin=pmin[params['ndim_x']-1-j])
                        #if ymax>pmax[params['ndim_x']-1-j]:
                        axis_corner[len(params['inf_pars'])-1-j,i].set_ylim(top=ymax*1.2)
                        if xmin<usepar_priormin[i]:
                            axis_corner[len(params['inf_pars'])-1-j,i].set_xlim(left=usepar_priormin[i])
                        if xmax>usepar_priormax[i]:
                            axis_corner[len(params['inf_pars'])-1-j,i].set_xlim(right=usepar_priormax[i])

                    # Make scatter plots on off-diagonal
                    else:
                        # Load contour info if it already exists
                        if self.params['load_plot_data'] == True:
                            vi_contour_info = [np.array(hf['contour_info']['testsamp%d' % r]['vi_Q_%d-%d_contours' % (len(params['inf_pars'])-1-j,i)]),
                                               np.array(hf['contour_info']['testsamp%d' % r]['vi_X_%d-%d_contours' % (len(params['inf_pars'])-1-j,i)]),
                                               np.array(hf['contour_info']['testsamp%d' % r]['vi_Y_%d-%d_contours' % (len(params['inf_pars'])-1-j,i)]),
                                               np.array(hf['contour_info']['testsamp%d' % r]['vi_L_%d-%d_contours' % (len(params['inf_pars'])-1-j,i)])]
                            bilby_contour_info = [np.array(hf['contour_info']['testsamp%d' % r]['bilby_Q_%d-%d_contours' % (len(params['inf_pars'])-1-j,i)]),
                                               np.array(hf['contour_info']['testsamp%d' % r]['bilby_Q_%d-%d_contours' % (len(params['inf_pars'])-1-j,i)]),
                                               np.array(hf['contour_info']['testsamp%d' % r]['bilby_Q_%d-%d_contours' % (len(params['inf_pars'])-1-j,i)]),
                                               np.array(hf['contour_info']['testsamp%d' % r]['bilby_Q_%d-%d_contours' % (len(params['inf_pars'])-1-j,i)])]
                        else:
                            vi_contour_info = None
                            bilby_contour_info = None

#                        axis_corner[params['ndim_x']-1-j,i].scatter(set1[i,:], set1[params['usepars'][::-1][j],:],c='r',s=0.2,alpha=0.5, label='VItamin')
#                        axis_corner[params['ndim_x']-1-j,i].scatter(set2[i,:], set2[params['usepars'][::-1][j],:],c='b',s=0.2,alpha=0.5, label=sampler)

#                        usepar_priormin = np.array(params['prior_min'])[params['usepars']]
#                        usepar_priormax = np.array(params['prior_max'])[params['usepars']]
#                        usepar_priormin[1] = 0.65000000000000000
#                        usepar_priormax[1] = 0.85000000000000000
#                        minimum_prior = [usepar_priormin[i],usepar_priormin[usepars_order[::-1][j]]]
#                        maximum_prior = [usepar_priormax[i],usepar_priormax[usepars_order[::-1][j]]]
                        minimum_prior = [usepar_priormin[i],usepar_priormin[::-1][j]]
                        maximum_prior = [usepar_priormax[i],usepar_priormax[::-1][j]]
                        comb_set1 = np.array([set1[i,:],set1[::-1,:][j,:]])
                        try:
                            cont1_out = self.make_contour_plot(axis_corner[len(params['inf_pars'])-1-j,i],set1[i,:],set1[::-1,:][j,:],
                                                               comb_set1,[parname1,parname2],minimum_prior,maximum_prior,'red',
                                                               self.params['load_plot_data'],vi_contour_info)

                            if self.params['load_plot_data'] == False:
                                # Save contour calculations in h5py files
                                testsamp_group.create_dataset('vi_Q_%d-%d_contours' % (len(params['inf_pars'])-1-j,i), data=cont1_out[0])
                                testsamp_group.create_dataset('vi_X_%d-%d_contours' % (len(params['inf_pars'])-1-j,i), data=cont1_out[1])
                                testsamp_group.create_dataset('vi_Y_%d-%d_contours' % (len(params['inf_pars'])-1-j,i), data=cont1_out[2])
                                testsamp_group.create_dataset('vi_L_%d-%d_contours' % (len(params['inf_pars'])-1-j,i), data=cont1_out[3])
                        except Exception as e:
                            print(e)
                            pass
                        comb_set2 = np.array([set2[i,:],set2[::-1,:][j,:]])

                        try:
                            cont2_out = self.make_contour_plot(axis_corner[len(params['inf_pars'])-1-j,i],set2[i,:],set2[::-1,:][j,:],
                                                               comb_set2,[parname1,parname2],minimum_prior,maximum_prior,'blue',
                                                               self.params['load_plot_data'],bilby_contour_info)
                            if self.params['load_plot_data'] == False:
                                testsamp_group.create_dataset('bilby_Q_%d-%d_contours' % (len(params['inf_pars'])-1-j,i), data=cont2_out[0])
                                testsamp_group.create_dataset('bilby_X_%d-%d_contours' % (len(params['inf_pars'])-1-j,i), data=cont2_out[1])
                                testsamp_group.create_dataset('bilby_Y_%d-%d_contours' % (len(params['inf_pars'])-1-j,i), data=cont2_out[2])
                                testsamp_group.create_dataset('bilby_L_%d-%d_contours' % (len(params['inf_pars'])-1-j,i), data=cont2_out[3])
                        except Exception as e:
                            print(e)
                            pass

                        # Add scalar truth crosshair
                        axis_corner[len(params['inf_pars'])-1-j,i].plot(pos_test_copy[r,i],pos_test_copy[r,::-1][j],'+k',markersize=6, label='Truth')

                        print('Made dataset')

                        xmin, xmax = axis_corner[len(params['inf_pars'])-1-j,i].get_xlim()
                        ymin, ymax = axis_corner[len(params['inf_pars'])-1-j,i].get_ylim()
                        if ymin<usepar_priormin[len(params['inf_pars'])-1-j]:
                            axis_corner[len(params['inf_pars'])-1-j,i].set_ylim(ymin=usepar_priormin[len(params['inf_pars'])-1-j])
                        if ymax>usepar_priormax[len(params['inf_pars'])-1-j]:
                            axis_corner[len(params['inf_pars'])-1-j,i].set_ylim(ymax=usepar_priormax[len(params['inf_pars'])-1-j])
                        if xmin<usepar_priormin[i]:
                            axis_corner[len(params['inf_pars'])-1-j,i].set_xlim(xmin=usepar_priormin[i])
                        if xmax>usepar_priormax[i]:
                            axis_corner[len(params['inf_pars'])-1-j,i].set_xlim(xmax=usepar_priormax[i])

                        if i==0 and j==1:
                            xtemp = np.array([0,100])
                            y1temp = np.array([0,100])
                            y2temp = y1temp + 100
                            axis_corner[len(params['inf_pars'])-1-j,i].fill_between(xtemp, y1temp, y2temp, facecolor='gray', interpolate=True,zorder=100)

                    axis_corner[len(params['inf_pars'])-1-j,i].grid(False)

                    axis_corner[len(params['inf_pars'])-1-j,i].tick_params(labelrotation=45,labelsize=8.0,axis='x')
                    axis_corner[len(params['inf_pars'])-1-j,i].tick_params(labelsize=8.0,axis='y')


                    # add labels
                    if i == 0 and len(params['inf_pars'])-1-j != 0:
                        axis_corner[len(params['inf_pars'])-1-j,i].set_ylabel(r"$%s$" % parname2,fontsize=9)
                        #axis_corner[params['ndim_x']-1-j,i].tick_params(axis="x", labelsize=12)
                        axis_corner[len(params['inf_pars'])-1-j,i].tick_params(axis="y", labelsize=9)
                    if len(params['inf_pars'])-1-j == (len(params['inf_pars'])-1):
                        axis_corner[len(params['inf_pars'])-1-j,i].set_xlabel(r"$%s$" % parname1,fontsize=9)
                        axis_corner[len(params['inf_pars'])-1-j,i].tick_params(axis="x", labelsize=9)
                    if i != 0:
                        # Turn off some some tick marks
                        axis_corner[len(params['inf_pars'])-1-j,i].yaxis.set_visible(False)
                    if i == 0 and len(params['inf_pars'])-1-j == 0:
                        axis_corner[len(params['inf_pars'])-1-j,i].yaxis.set_visible(False)
                    if len(params['inf_pars'])-1-j != len(params['inf_pars'])-1:
                        axis_corner[len(params['inf_pars'])-1-j,i].xaxis.set_visible(False)

                    # set tick labels to not use scientific notation
                    axis_corner[len(params['inf_pars'])-1-j,i].ticklabel_format(axis='both',useOffset=False,style='plain')

                    # Remove whitespace on x-axis in all plots
                    axis_corner[len(params['inf_pars'])-1-j,i].margins(x=0,y=0)

                tmp_idx -= 1

            tmp_idx = len(params['inf_pars'])

            for i in range(len(params['inf_pars'])-1):
                for j in range(tmp_idx-1):
                    axis_corner[i,j+i+1].set_axis_off()
                tmp_idx -= 1

            # remove subplots not used 
#            axis_corner[0,1].set_axis_off()
#            axis_corner[0,2].set_axis_off()
#            axis_corner[1,2].set_axis_off()

            # plot corner plot
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.autoscale(tight=True)

            tmp_idx=len(params['inf_pars'])
            fig_corner.align_ylabels(axis_corner[:, :])
            fig_corner.align_xlabels(axis_corner[:, :])
            #fig_corner.canvas.draw()
            fig_corner.savefig('%s/output_%s_test%d_%d.png' % (params['plot_dir'],params['run_label'],test_sample_idx,epoch_idx),dpi=360,bbox_inches='tight',pad_inches=0)
            fig_corner.savefig('%s/latest_%s/output_%s_test%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],test_sample_idx),dpi=360,bbox_inches='tight',pad_inches=0)
            plt.close(fig_corner)
            plt.close('all')

        hf.close()
        print('Finished corner plotting routine')
        del set2
        del set1

        return

    def make_overlap_plot(self,epoch,iterations,s,olvec,olvec_2d,adksVec):
        olvec_1d = np.zeros((self.params['r'],self.params['r'],self.params['ndim_x']))
        fig, axes = plt.subplots(1,1,figsize=(6,6))

        cnt_2d=0
        # Make 2D scatter plots of posteriors
        for k in range(self.params['ndim_x']):
            parname1 = self.params['parnames'][k]
            for nextk in range(self.params['ndim_x']):
                parname2 = self.params['parnames'][nextk]
                if nextk>k:
                    cnt = 0

                    # initialize 2D plot for showing testing results
                    fig, axes = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),sharex='all',sharey='all',squeeze=False)

                    # initialize 1D plots for showing testing results
                    fig_1d, axes_1d = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),squeeze=False)

                    # initialize 1D plots for showing testing results for last 1d hist
                    fig_1d_last, axes_1d_last = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),squeeze=False)

                    # initialize 1D plots for showing testing results
                    #fig_kl, axis_kl = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6))
                    fig_ad, axis_ad = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),squeeze=False)
                    fig_ks, axis_ks = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),squeeze=False)

                    # initialize 1D plots for showing testing results for last 1d hist
                    #fig_kl_last, axis_kl_last = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6))
                    fig_ad_last, axis_ad_last = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),squeeze=False)
                    fig_ks_last, axis_ks_last = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),squeeze=False)

                    # Iterate over test cases
                    for i in range(self.params['r']):
                        for j in range(self.params['r']):

                            # remove samples outside of the prior mass distribution
                            mask = [(self.rev_x[cnt,0,:] >= self.rev_x[cnt,2,:]) & (self.rev_x[cnt,3,:] >= 0.0) & (self.rev_x[cnt,3,:] <= 1.0) & (self.rev_x[cnt,1,:] >= 0.0) & (self.rev_x[cnt,1,:] <= 1.0) & (self.rev_x[cnt,0,:] >= 0.0) & (self.rev_x[cnt,0,:] <= 1.0) & (self.rev_x[cnt,2,:] <= 1.0) & (self.rev_x[cnt,2,:] >= 0.0)]
                            mask = np.argwhere(mask[0])
                            new_rev = self.rev_x[cnt,nextk,mask]
                            new_rev = new_rev.reshape(new_rev.shape[0])
                            new_samples = self.samples[cnt,mask,k]
                            new_samples = new_samples.reshape(new_samples.shape[0])

                            # compute the n-d overlap
                            if k==0 and nextk==1:
                                ol = data_maker.overlap(self.samples[cnt,:,:],self.rev_x[cnt,:,:].T,next_cnt=True)
                                olvec[i,j,s] = ol

                                # print A-D and K-S test
                                ks_mcmc_arr, ks_inn_arr, ad_mcmc_arr, ad_inn_arr, kl_mcmc_arr, kl_inn_arr = self.ad_ks_test(self.params['parnames'],self.rev_x[cnt,self.params['usepars'],:], self.samples[cnt,:,:self.params['ndim_x']], cnt)
                                for p in self.params['usepars']:
                                    for c in range(6):
                                        for w in range(self.params['n_kl_samp']):
                                            adksVec[i,j,p,c,w] = np.array([ks_mcmc_arr,ks_inn_arr,ad_mcmc_arr,ad_inn_arr,kl_mcmc_arr, kl_inn_arr])[c,p,w]

                            # get 2d overlap values
                            #samples_2d = np.array([self.samples[cnt,:,k],self.samples[cnt,:,nextk]]).T
                            #rev_x_2d = np.array([self.rev_x[cnt,k,:],self.rev_x[cnt,nextk,:]]).T
                            #ol_2d = data_maker.overlap(self.samples[cnt,:,:],self.rev_x[cnt,:,:],k,nextk)
                            #olvec_2d[i,j,s,cnt_2d] = ol_2d

                            # plot the samples and the true contours
                            axes[i,j].clear()
                            axes[i,j].scatter(self.rev_x[cnt,k,mask].reshape(mask.shape[0]), self.rev_x[cnt,nextk,mask].reshape(mask.shape[0]),c='r',s=0.2,alpha=0.5, label='VICI')
                            axes[i,j].scatter(self.samples[cnt,mask,k].reshape(mask.shape[0]), self.samples[cnt,mask,nextk].reshape(mask.shape[0]),c='b',s=0.2,alpha=0.5, label='MCMC')
                            #axes[i,j].set_xlim([0,1])
                            #axes[i,j].set_ylim([0,1])
                            axes[i,j].plot(self.pos_test[cnt,k],self.pos_test[cnt,nextk],'+c',markersize=8, label='Truth')
                            oltxt = '%.2f' % olvec[i,j,s]
                            axes[i,j].text(0.90, 0.95, oltxt,
                                horizontalalignment='right',
                                verticalalignment='top',
                                    transform=axes[i,j].transAxes)
                            matplotlib.rc('xtick', labelsize=8)
                            matplotlib.rc('ytick', labelsize=8)
                            axes[i,j].set_xlabel(parname1,fontsize=14) if i==self.params['r']-1 else axes[i,j].set_xlabel('')
                            axes[i,j].set_ylabel(parname2,fontsize=14) if j==0 else axes[i,j].set_ylabel('')
                            if i == 0 and j == 0: axes[i,j].legend(loc='upper left', fontsize='x-small')

                            def confidence_bd(samp_array):
                                """
                                compute confidence bounds for a given array
                                """
                                cf_bd_sum_lidx = 0
                                cf_bd_sum_ridx = 0
                                cf_bd_sum_left = 0
                                cf_bd_sum_right = 0
                                cf_perc = 0.05

                                cf_bd_sum_lidx = np.sort(samp_array)[int(len(samp_array)*cf_perc)]
                                cf_bd_sum_ridx = np.sort(samp_array)[int(len(samp_array)*(1.0-cf_perc))]

                                return [cf_bd_sum_lidx, cf_bd_sum_ridx]

                            # plot the 1D samples and the 5% confidence bounds
                            ol_hist = data_maker.overlap(self.samples[cnt,mask,k].reshape(mask.shape[0],1),self.rev_x[cnt,k,mask].reshape(mask.shape[0],1),k)
                            olvec_1d[i,j,k] = ol_hist
                            n_hist_bins=30
                            axes_1d[i,j].clear()
                            axes_1d[i,j].hist(self.samples[cnt,mask,k].reshape(mask.shape[0]),color='b',bins=n_hist_bins,alpha=0.5,density=True)
                            axes_1d[i,j].hist(self.rev_x[cnt,k,mask].reshape(mask.shape[0]),color='r',bins=n_hist_bins,alpha=0.5,density=True)
                            for xtick,ytick in zip(axes_1d[i,j].xaxis.get_major_ticks(),axes_1d[i,j].yaxis.get_major_ticks()):
                                    xtick.label.set_fontsize(4)
                                    ytick.label.set_fontsize(4)
                            #axes_1d[i,j].set_xlim([0,1])
                            axes_1d[i,j].axvline(x=self.pos_test[cnt,k], linewidth=0.5, color='black')
                            axes_1d[i,j].axvline(x=confidence_bd(self.samples[cnt,mask,k].reshape(mask.shape[0]))[0], linewidth=0.5, color='b')
                            axes_1d[i,j].axvline(x=confidence_bd(self.samples[cnt,mask,k].reshape(mask.shape[0]))[1], linewidth=0.5, color='b')
                            axes_1d[i,j].axvline(x=confidence_bd(self.rev_x[cnt,k,mask].reshape(mask.shape[0]))[0], linewidth=0.5, color='r')
                            axes_1d[i,j].axvline(x=confidence_bd(self.rev_x[cnt,k,mask].reshape(mask.shape[0]))[1], linewidth=0.5, color='r')
                            oltxt = '%.2f' % olvec_1d[i,j,k]
                            axes_1d[i,j].text(0.90, 0.95, oltxt,
                                horizontalalignment='right',
                                verticalalignment='top',
                                    transform=axes_1d[i,j].transAxes)
                            #matplotlib.rc('xtick', labelsize=4)
                            #matplotlib.rc('ytick', labelsize=4)
                            axes_1d[i,j].set_xlabel(parname1,fontsize=14) if i==self.params['r']-1 else axes_1d[i,j].set_xlabel('')

                            # Plot statistic histograms
                            try:
                                axis_ks[i,j].hist(adksVec[i,j,k,0,:],bins=n_hist_bins,alpha=0.5,color='blue',normed=True,label='Bilby')
                                axis_ks[i,j].hist(adksVec[i,j,k,1,:],bins=n_hist_bins,alpha=0.5,color='red',normed=True,label='VICI')
                                for xtick,ytick in zip(axis_ks[i,j].xaxis.get_major_ticks(),axis_ks[i,j].yaxis.get_major_ticks()):
                                    xtick.label.set_fontsize(4)
                                    ytick.label.set_fontsize(4)

                                axis_ad[i,j].hist(adksVec[i,j,k,2,:],bins=n_hist_bins,alpha=0.5,color='blue',normed=True,label='Bilby')
                                axis_ad[i,j].hist(adksVec[i,j,k,3,:],bins=n_hist_bins,alpha=0.5,color='red',normed=True,label='VICI')
                                for xtick,ytick in zip(axis_ad[i,j].xaxis.get_major_ticks(),axis_ad[i,j].yaxis.get_major_ticks()):
                                    xtick.label.set_fontsize(4)
                                    ytick.label.set_fontsize(4)

                                # normalize k-l results
                                #kl_max = np.max([adksVec[i,j,k,4,:],adksVec[i,j,k,5,:]])
                                #adksVec[i,j,k,4,:] = adksVec[i,j,k,4,:] / kl_max
                                #adksVec[i,j,k,5,:] = adksVec[i,j,k,5,:] / kl_max
                                #axis_kl[i,j].hist(adksVec[i,j,k,4,:],bins=n_hist_bins,alpha=0.5,color='blue',normed=True,label='Bilby')
                                #axis_kl[i,j].hist(adksVec[i,j,k,5,:],bins=n_hist_bins,alpha=0.5,color='red',normed=True,label='VICI')
                                #for xtick,ytick in zip(axis_kl[i,j].xaxis.get_major_ticks(),axis_kl[i,j].yaxis.get_major_ticks()):
                                #    xtick.label.set_fontsize(4)
                                #    ytick.label.set_fontsize(4)

                                #axis_kl[i,j].set_xlabel('KL Values') if i==self.params['r']-1 else axis_kl[i,j].set_xlabel('')
                                axis_ks[i,j].set_xlabel('ks-stat') if i==self.params['r']-1 else axis_ks[i,j].set_xlabel('')
                                axis_ad[i,j].set_xlabel('ad-stat') if i==self.params['r']-1 else axis_ad[i,j].set_xlabel('')

                            except IndexError:
                                print('Warning: bad stat result!')
                                continue

                            if i == 0 and j == 0: 
                                #axis_kl[i,j].legend(loc='upper left', fontsize='x-small')
                                axis_ad[i,j].legend(loc='upper left', fontsize='x-small')
                                axis_ks[i,j].legend(loc='upper left', fontsize='x-small')

                            if k == (self.params['ndim_x']-2):
                                # plot the 1D samples and the 5% confidence bounds
                                ol_hist = data_maker.overlap(self.samples[cnt,mask,k+1].reshape(mask.shape[0],1),self.rev_x[cnt,k+1,mask].reshape(mask.shape[0],1),k)
                                olvec_1d[i,j,k+1] = ol_hist
                                axes_1d_last[i,j].clear()
                                axes_1d_last[i,j].hist(self.samples[cnt,mask,k+1].reshape(mask.shape[0]),color='b',bins=n_hist_bins,alpha=0.5,density=True)
                                axes_1d_last[i,j].hist(self.rev_x[cnt,k+1,mask].reshape(mask.shape[0]),color='r',bins=n_hist_bins,alpha=0.5,density=True)
                                for xtick,ytick in zip(axes_1d_last[i,j].xaxis.get_major_ticks(),axes_1d_last[i,j].yaxis.get_major_ticks()):
                                    xtick.label.set_fontsize(4)
                                    ytick.label.set_fontsize(4)
                                #axes_1d_last[i,j].set_xlim([0,1])
                                axes_1d_last[i,j].axvline(x=self.pos_test[cnt,k+1], linewidth=0.5, color='black')
                                axes_1d_last[i,j].axvline(x=confidence_bd(self.samples[cnt,mask,k+1].reshape(mask.shape[0]))[0], linewidth=0.5, color='b')
                                axes_1d_last[i,j].axvline(x=confidence_bd(self.samples[cnt,mask,k+1].reshape(mask.shape[0]))[1], linewidth=0.5, color='b')
                                axes_1d_last[i,j].axvline(x=confidence_bd(self.rev_x[cnt,k+1,mask].reshape(mask.shape[0]))[0], linewidth=0.5, color='r')
                                axes_1d_last[i,j].axvline(x=confidence_bd(self.rev_x[cnt,k+1,mask].reshape(mask.shape[0]))[1], linewidth=0.5, color='r')
                                oltxt = '%.2f' % olvec_1d[i,j,k+1]
                                axes_1d_last[i,j].text(0.90, 0.95, oltxt,
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                        transform=axes_1d_last[i,j].transAxes)
                                #matplotlib.rc('xtick', labelsize=4)
                                #matplotlib.rc('ytick', labelsize=4)
                                axes_1d_last[i,j].set_xlabel(self.params['parnames'][k+1]) if i==self.params['r']-1 else axes_1d_last[i,j].set_xlabel('')

                                # Plot statistic histograms
                                if self.params['do_adkskl_test']:
                                    try:
                                        axis_ks_last[i,j].hist(adksVec[i,j,k+1,0,:],bins=n_hist_bins,alpha=0.5,color='blue',normed=True)
                                        axis_ks_last[i,j].hist(adksVec[i,j,k+1,1,:],bins=n_hist_bins,alpha=0.5,color='red',normed=True)
                                        for xtick,ytick in zip(axis_ks_last[i,j].xaxis.get_major_ticks(),axis_ks_last[i,j].yaxis.get_major_ticks()):
                                            xtick.label.set_fontsize(4)
                                            ytick.label.set_fontsize(4)

                                        axis_ad_last[i,j].hist(adksVec[i,j,k+1,2,:],bins=n_hist_bins,alpha=0.5,color='blue',normed=True)
                                        axis_ad_last[i,j].hist(adksVec[i,j,k+1,3,:],bins=n_hist_bins,alpha=0.5,color='red',normed=True)
                                        for xtick,ytick in zip(axis_ad_last[i,j].xaxis.get_major_ticks(),axis_ad_last[i,j].yaxis.get_major_ticks()):
                                            xtick.label.set_fontsize(4)
                                            ytick.label.set_fontsize(4)

                                        #axis_kl_last[i,j].hist(adksVec[i,j,k+1,4,:],bins=n_hist_bins,alpha=0.5,color='blue',normed=True)
                                        #axis_kl_last[i,j].hist(adksVec[i,j,k+1,5,:],bins=n_hist_bins,alpha=0.5,color='red',normed=True)
                                        #for xtick,ytick in zip(axis_kl_last[i,j].xaxis.get_major_ticks(),axis_kl_last[i,j].yaxis.get_major_ticks()):
                                        #    xtick.label.set_fontsize(4)
                                        #    ytick.label.set_fontsize(4)

                                        #axis_kl[i,j].set_xlabel('KL Values') if i==self.params['r']-1 else axis_kl[i,j].set_xlabel('')
                                        axis_ks_last[i,j].set_xlabel('ks-stat') if i==self.params['r']-1 else axis_ks[i,j].set_xlabel('')
                                        axis_ad_last[i,j].set_xlabel('ad-stat') if i==self.params['r']-1 else axis_ad[i,j].set_xlabel('')
                                    except IndexError:
                                        print('Warning: bad stat result!')
                                        continue

                            cnt += 1

                        # save the results to file
                        fig_1d.canvas.draw()
                        fig_1d.savefig('%s/latest/latest-1d_%d.png' % (self.params['plot_dir'][0],k),dpi=360)

                        if self.params['do_adkskl_test']:
                            #fig_kl.canvas.draw()
                            #fig_kl.savefig('%s/latest/hist-kl_%d.png' % (self.params['plot_dir'][0],k),dpi=360)
                            fig_ad.canvas.draw()
                            fig_ad.savefig('%s/latest/hist-ad_%d.png' % (self.params['plot_dir'][0],k),dpi=360)
                            fig_ks.canvas.draw()
                            fig_ks.savefig('%s/latest/hist-ks_%d.png' % (self.params['plot_dir'][0],k),dpi=360)
                            #plt.close(fig_kl)
                            plt.close(fig_ks)
                            plt.close(fig_ad)
                        if k == (self.params['ndim_x']-2):
                            # save the results to file
                            fig_1d_last.canvas.draw()
                            fig_1d_last.savefig('%s/latest/latest-1d_%d.png' % (self.params['plot_dir'][0],k+1),dpi=360)

                            if self.params['do_adkskl_test']:
                                fig_ks_last.canvas.draw()
                                fig_ks_last.savefig('%s/latest/hist-ks_%d.png' % (self.params['plot_dir'][0],k+1),dpi=360)
                                fig_ad_last.canvas.draw()
                                fig_ad_last.savefig('%s/latest/hist-ad_%d.png' % (self.params['plot_dir'][0],k+1),dpi=360)
                                #fig_kl_last.canvas.draw()
                                #fig_kl_last.savefig('%s/latest/hist-kl_%d.png' % (self.params['plot_dir'][0],k+1),dpi=360)
                                #plt.close(fig_kl_last)
                                plt.close(fig_ks_last)
                                plt.close(fig_ad_last)

                        # save the results to file
                        fig.canvas.draw()
                        fig.savefig('%s/latest/posteriors_%d%d.png' % (self.params['plot_dir'][0],k,nextk),dpi=360)
                        plt.close(fig)
                        cnt_2d+=1

        s+=1
        
        # plot overlap results
        fig, axes = plt.subplots(1,figsize=(6,6))
        plot_cadence=(self.params['num_iterations']-1)
        for i in range(self.params['r']):
            for j in range(self.params['r']):
                color = next(axes._get_lines.prop_cycler)['color']
                axes.semilogx(np.arange(epoch, step=plot_cadence),olvec[i,j,:int((epoch)/plot_cadence)],alpha=0.5, color=color)
                axes.plot([int(epoch)],[olvec[i,j,int(epoch/plot_cadence)]],'.', color=color)
        axes.grid()
        axes.set_ylabel('overlap')
        axes.set_xlabel('epoch')
        axes.set_ylim([0,1])
        plt.savefig('%s/latest/overlap_logscale.png' % self.params['plot_dir'], dpi=360)
        plt.close(fig)      

        fig, axes = plt.subplots(1,figsize=(6,6))
        for i in range(self.params['r']):
            for j in range(self.params['r']):
                color = next(axes._get_lines.prop_cycler)['color']
                axes.plot(np.arange(epoch, step=plot_cadence),olvec[i,j,:int((epoch)/plot_cadence)],alpha=0.5, color=color)
                axes.plot([int(epoch)],[olvec[i,j,int(epoch/plot_cadence)]],'.', color=color)
        axes.grid()
        axes.set_ylabel('overlap')
        axes.set_xlabel('epoch')
        axes.set_ylim([0,1])
        plt.savefig('%s/latest/overlap.png' % self.params['plot_dir'], dpi=360)
        plt.close(fig)

        plt.close('all')
        

        # plot ad and ks results [ks_mcmc_arr,ks_inn_arr,ad_mcmc_arr,ad_inn_arr]
        """ 
        for p in range(self.params['ndim_x']):
            fig_ks, axis_ks = plt.subplots(1,figsize=(6,6)) 
            fig_ad, axis_ad = plt.subplots(1,figsize=(6,6))
            for i in range(self.params['r']):
                for j in range(self.params['r']):
                    axis_ks[i,j].hist(adksVec[i,j,p,0,:],alpha=0.5,color='blue')
                    axis_ks[i,j].hist(adksVec[i,j,p,1,:],alpha=0.5,color='red')

                    axis_ad[i,j].hist(adksVec[i,j,p,2,:],alpha=0.5,color='blue')
                    axis_ad[i,j].hist(adksVec[i,j,p,3,:],alpha=0.5,color='red')
                    

            axis_ks.set_xlabel('Epoch')
            axis_ad.set_xlabel('Epoch')
            axis_ks.set_ylabel('KS Statistic')
            axis_ad.set_ylabel('AD Statistic')
            fig_ks.savefig('%s/latest/ks_%s_stat.png' % (self.params['plot_dir'][0],self.params['parnames'][p]), dpi=360)
            fig_ad.savefig('%s/latest/ad_%s_stat.png' % (self.params['plot_dir'][0],self.params['parnames'][p]), dpi=360)
            plt.close(fig_ks)
            plt.close(fig_ad)
        """
        return s, olvec, olvec_2d
