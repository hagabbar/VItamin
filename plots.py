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

class make_plots:
    """
    Class for making results plots
    """
    
    def __init__(self,params,samples,rev_x,pos_test):
        """
        Add variables here later if need be
        """
        self.params = params
        self.samples = samples
        self.rev_x = rev_x
        self.pos_test = pos_test

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
                    VI_pred, _, _, dt,_  = model.run(params, np.expand_dims(sig_test[i],axis=0), np.shape(par_test)[1],
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

                rand_idx_posterior = np.random.choice(np.linspace(0,XS.shape[0]-1,dtype=np.int),self.params['n_samples']) 
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

        # Store above declared functions to be used later
        self.load_test_set = load_test_set

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

    def plot_pp(self,model,sig_test,par_test,i_epoch,normscales,pos_test,bounds):
        """
        make p-p plots
        
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
                if self.params['n_filters_r1'] != None:
                    y = sig_test[cnt,:].reshape(1,sig_test.shape[1],sig_test.shape[2])
                else:
                    y = sig_test[cnt,:].reshape(1,sig_test.shape[1])
                 # The trained inverse model weights can then be used to infer a probability density of solutions 
#given new measurements
                x, _, _, dt,_  = model.run(self.params, y, np.shape(par_test)[1],
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
                axis.plot(np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0),np.sort(pp[:,j]),'-',color='red',linewidth=1,zorder=50,label=r'$\textrm{VItamin}$',alpha=0.5)
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
                    axis.plot(np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0),np.sort(pp_bilby),'-',color=CB_color_cycle[i-1],linewidth=1,label=r'$\textrm{%s}$' % samplers[i],alpha=0.5)
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
        for j in range(len(confidence)):
            axis.plot(np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0),confidence_bound[j,0,:],color='gray')
            axis.plot(np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0),confidence_bound[j,1,:],color='gray')
            axis.fill_between(np.arange((self.params['r']**2)+2)/((self.params['r']**2)+1.0), confidence_bound[j,0,:], confidence_bound[j,1,:], facecolor=conf_color_wheel[j], alpha=0.5)
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
        plt.semilogx(xvec,np.array(plotdata)[:,0],label=r'$\mathrm{Recon}$',color='blue',alpha=0.5)
        plt.semilogx(xvec,np.array(plotdata)[:,1],label=r'$\mathrm{KL}$',color='orange',alpha=0.5)
        plt.semilogx(xvec,np.array(plotdata)[:,2],label=r'$\mathrm{Total}$',color='green',alpha=0.5)
        plt.semilogx(xvec,np.array(plotdata)[:,3],color='blue',linestyle='dotted')
        plt.semilogx(xvec,np.array(plotdata)[:,4],color='orange',linestyle='dotted')
        plt.semilogx(xvec,np.array(plotdata)[:,5],color='green',linestyle='dotted')
        plt.ylim([-25,15])
        plt.xlabel(r'$\mathrm{Iteration}$')
        plt.ylabel(r'$\mathrm{Cost}$')
        plt.legend()
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
                    else:
                        kl_result_all[:,r] = 0   

                return kl_result_all
            else:
                kl_result = []
                set1 = set1.T
                set2 = set2.T
                for kl_idx in range(10):
                    rand_idx_kl = np.random.choice(np.linspace(0,set1.shape[0]-1,dtype=np.int),size=100)
                    kl_result.append(estimate(set1[rand_idx_kl,:],set2[rand_idx_kl,:]) + estimate(set2[rand_idx_kl,:],set1[rand_idx_kl,:]))
                kl_result = np.mean(kl_result)

                return kl_result
   
        # Define variables 
        params = self.params
        usesamps = params['samplers']
        samplers = params['samplers']
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
            hf = h5py.File('plotting_data_%s/KL_plot_data.h5' % params['run_label'], 'r')
     
        
        # 4 pannel KL approach
        fig_kl, axis_kl = plt.subplots(2,2,figsize=(6,6),sharey=True,sharex=True)
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
                        print_cnt+=1
                        sampler1, sampler2 = samplers[i]+'1', samplers[::-1][j]+'1'
                        # currently not doing KL of approaches with themselves, so skip here
                        continue
                    else:
                        sampler1, sampler2 = samplers[i]+'1', samplers[::-1][j]+'1'
                   
                        if self.params['load_plot_data'] == False:
                            set1,time1 = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=sampler1,vitamin_pred_made=vi_pred_made)
                            set2,time2 = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=sampler2,vitamin_pred_made=vi_pred_made)

                            # check if vitamin test posteriors were generated for the first time
                            if sampler1 == 'vitamin1' and vi_pred_made == None:
                                vi_pred_made = [set1,time1]
                            elif sampler2 == 'vitamin1' and vi_pred_made == None:
                                vi_pred_made = [set2,time2]

                    if self.params['load_plot_data'] == True:
                        tot_kl = np.array(hf['%s-%s' % (sampler1,sampler2)])
                    else:
                        # Iterate over test cases
                        tot_kl = []  # total KL over all infered parameters

                        for r in range(self.params['r']**2):
                            tot_kl.append(compute_kl(set1[r],set2[r],[sampler1,sampler2]))
                            print('Completed KL for set %s-%s and test sample %s' % (sampler1,sampler2,str(r)))
                        tot_kl = np.array(tot_kl)

                    if self.params['load_plot_data'] == False:
                        # Save results to h5py file
                        hf.create_dataset('%s-%s' % (sampler1,sampler2), data=tot_kl)

                    logbins = np.logspace(-3,2.5,50)
                    logbins_indi = np.logspace(-3,3,50)

                    # plot colored hist
                    if samplers[i] == 'vitamin' and samplers[::-1][j] == samplers[1:][k]: 
                        print(tot_kl.argsort()[-15:][::-1])
                        print(np.sort(tot_kl)[-15:][::-1])
                        print(tot_kl.argsort()[:15][:])
                        print(np.sort(tot_kl)[:15][:])
                        axis_kl[kl_idx_1,kl_idx_2].hist(tot_kl,log=True,bins=logbins,alpha=0.5,histtype='stepfilled',density=True,color=CB_color_cycle[print_cnt],label='$\mathrm{%s \ vs. \ %s}$' % (samplers[i],samplers[::-1][j]),zorder=2)
                        axis_kl[kl_idx_1,kl_idx_2].hist(tot_kl,log=True,bins=logbins,histtype='step',density=True,facecolor='None',ls='-',lw=2,edgecolor=CB_color_cycle[print_cnt],zorder=10)
                    # record non-colored hists
                    elif samplers[i] != 'vitamin' and samplers[::-1][j] != 'vitamin':
                        if samplers[i] == samplers[1:][k] or samplers[::-1][j] == samplers[1:][k]:
                            print(tot_kl.argsort()[-15:][::-1])
                            print(np.sort(tot_kl)[-15:][::-1])
                            print(tot_kl.argsort()[:15][:])
                            print(np.sort(tot_kl)[:15][:])

                            tot_kl_grey = np.append(tot_kl_grey,tot_kl)

                            print('Mean total KL between bilby samps: %s' % str(np.mean(tot_kl)))
                    print('Completed KL calculation %d/%d' % (print_cnt,len(usesamps)*2))
                    print_cnt+=1
                tmp_idx-=1

            # Plot non-colored histograms
            axis_kl[kl_idx_1,kl_idx_2].hist(np.array(tot_kl_grey).squeeze(),log=True,bins=logbins,alpha=0.8,histtype='stepfilled',density=True,color='grey',label=r'$\mathrm{%s \ vs. \ other \ samplers}$' % samplers[1:][k],zorder=1)
            axis_kl[kl_idx_1,kl_idx_2].hist(np.array(tot_kl_grey).squeeze(),log=True,bins=logbins,histtype='step',density=True,facecolor='None',ls='-',lw=2,edgecolor='grey',zorder=1)

            # plot KL histograms
            if kl_idx_1 == 1:
                axis_kl[kl_idx_1,kl_idx_2].set_xlabel(r'$\mathrm{KL-Statistic}$',fontsize=14)
            if kl_idx_2 == 0:
                axis_kl[kl_idx_1,kl_idx_2].set_ylabel(r'$p(\mathrm{KL})$',fontsize=14)
            axis_kl[kl_idx_1,kl_idx_2].tick_params(axis="both", labelsize=12, direction='out')
            leg = axis_kl[kl_idx_1,kl_idx_2].legend(loc='upper right', fontsize=4) #'medium')
            for l in leg.legendHandles:
                l.set_alpha(1.0)

            axis_kl[kl_idx_1,kl_idx_2].set_xlim(left=8e-2,right=100)
            axis_kl[kl_idx_1,kl_idx_2].set_xscale('log')
            axis_kl[kl_idx_1,kl_idx_2].set_yscale('log')
            axis_kl[kl_idx_1,kl_idx_2].grid(False)
            print()
            print('Made hist plot %d' % k)
            print()

        # Save figure
        fig_kl.canvas.draw()
        fig_kl.savefig('%s/latest_%s/hist-kl.png' % (self.params['plot_dir'],self.params['run_label']),dpi=360)
        plt.close(fig_kl)
        hf.close()
 
        return
