import os, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np
from scipy.stats import uniform, norm, gaussian_kde, ks_2samp, anderson_ksamp
from scipy import stats
import h5py

from data import chris_data as data_maker
from Models import VICI_inverse_model
from Models import CVAE

def make_dirs(out_dir):
    """
    Make directories to store plots. Directories that already exist will be overwritten.
    """

    ## If file exists, delete it ##
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    else:    ## Show a message ##
        print("Attention: %s file not found" % out_dir)

    # setup output directory - if it does not exist
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
            kl_mcmc_arr = []
            kl_inn_arr = []

            cur_max = self.params['n_samples']
            mcmc = []
            c=vici = []
            for i in range(inn_samps.shape[0]):
                # remove samples outside of the prior mass distribution
                mask = [(inn_samps[0,:] >= inn_samps[2,:]) & (inn_samps[3,:] >= 1000.0) & (inn_samps[3,:] <= 3000.0) & (inn_samps[1,:] >= 0.4) & (inn_samps[1,:] <= 0.6) & (inn_samps[0,:] >= 35.0) & (inn_samps[0,:] <= 80.0) & (inn_samps[2,:] <= 80.0) & (inn_samps[2,:] >= 35.0)]
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
                kl_mcmc_samps = []
                kl_inn_samps = []
                n_samps = self.params['n_samples']
                n_pars = self.params['ndim_x']

                # iterate over number of randomized sample slices
                for j in range(self.params['n_kl_samp']):
                    # get ideal bayesian number. We want the 2 tailed p value from the KS test FYI
                    ks_mcmc_result = ks_2samp(np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0)), np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0)))
                    ad_mcmc_result = anderson_ksamp([np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0)), np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0))])
                    kl_mcmc_result = np.sum(mcmc[:,np.random.randint(0,high=int(mcmc.shape[1]),size=int(mcmc.shape[1]/2.0))] * ( np.log(mcmc[:,np.random.randint(0,high=int(mcmc.shape[1]),size=int(mcmc.shape[1]/2.0))])
                                            - np.log(mcmc[:,np.random.randint(0,high=int(mcmc.shape[1]),size=int(mcmc.shape[1]/2.0))]) ))
                

                    # get predicted vs. true number
                    ks_inn_result = ks_2samp(np.random.choice(vici[i,:],size=int(mcmc.shape[1]/2.0)),np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0)))
                    ad_inn_result = anderson_ksamp([np.random.choice(vici[i,:],size=int(mcmc.shape[1]/2.0)),np.random.choice(mcmc[i,:],size=int(mcmc.shape[1]/2.0))])
                    kl_inn_result = np.sum(mcmc[:,np.random.randint(0,high=int(mcmc.shape[1]),size=int(mcmc.shape[1]/2.0))] * ( np.log(mcmc[:,np.random.randint(0,high=int(mcmc.shape[1]),size=int(mcmc.shape[1]/2.0))]) 
                                            - np.log(vici[:,np.random.randint(0,high=int(mcmc.shape[1]),size=int(mcmc.shape[1]/2.0))]) ))

                    # store result stats
                    ks_mcmc_samps.append(ks_mcmc_result[1])
                    ks_inn_samps.append(ks_inn_result[1])
                    ad_mcmc_samps.append(ad_mcmc_result[0])
                    ad_inn_samps.append(ad_inn_result[0])
                    kl_mcmc_samps.append(kl_mcmc_result)
                    kl_inn_samps.append(kl_inn_result)
                print('Test Case %d, Parameter(%s) k-s result: [Ideal(%.6f), Predicted(%.6f)]' % (int(cnt),parnames[i],np.array(ks_mcmc_result[1]),np.array(ks_inn_result[1])))
                print('Test Case %d, Parameter(%s) A-D result: [Ideal(%.6f), Predicted(%.6f)]' % (int(cnt),parnames[i],np.array(ad_mcmc_result[0]),np.array(ad_inn_result[0])))
                print('Test Case %d, Parameter(%s) K-L result: [Ideal(%.6f), Predicted(%.6f)]' % (int(cnt),parnames[i],np.array(kl_mcmc_result),np.array(kl_inn_result)))

                # store result stats
                ks_mcmc_arr.append(ks_mcmc_samps)
                ks_inn_arr.append(ks_inn_samps)
                ad_mcmc_arr.append(ad_mcmc_samps)
                ad_inn_arr.append(ad_inn_samps)
                kl_mcmc_arr.append(kl_mcmc_samps)
                kl_inn_arr.append(kl_inn_samps)

            return ks_mcmc_arr, ks_inn_arr, ad_mcmc_arr, ad_inn_arr, kl_mcmc_arr, kl_inn_arr

        self.ad_ks_test = ad_ks_test

    def pp_plot(self,truth,samples):
        """
        generates the pp plot data given samples and truth values
        """
        Nsamp = samples.shape[0]
        kernel = gaussian_kde(samples.transpose())
        v = kernel.pdf(truth)
        x = kernel.pdf(samples.transpose())
        r = np.sum(x>v)/float(Nsamp)

        return r

    def plot_pp(self,model,sig_test,par_test,i_epoch,normscales):
        """
        make p-p plots
        """
        Npp = self.params['Npp']
        Nsamp = 1#self.params['n_samples']
        ndim_y = self.params['ndata']
        outdir = self.params['plot_dir'][0]
        
        
        plt.figure()
        pp = np.zeros(Npp+2)
        pp[0] = 0.0
        pp[1] = 1.0
        for cnt in range(Npp):

            y = np.tile(np.array(sig_test[cnt,:]),Nsamp).reshape(Nsamp,ndim_y)

            # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
            _, _, x, _ = model.run(self.params, y, np.shape(par_test)[1], "inverse_model_dir_%s/inverse_model.ckpt" % self.params['run_label']) # This runs the trained model using the weights stored in inverse_model_dir/inverse_model.ckpt

            # Convert XS back to unnormalized version
            if self.params['do_normscale']:
                for m in range(self.params['ndim_x']):
                    x[:,m,:] = x[:,m,:]*normscales[m]

            x = x.reshape(x.shape[2],x.shape[1])
            pp[cnt+2] = self.pp_plot(par_test[cnt,:],x[:,:])
            print('Computed p-p plot iteration %d/%d' % (int(cnt),int(Npp)))

        plt.plot(np.arange(Npp+2)/(Npp+1.0),np.sort(pp),'-')
        plt.plot([0,1],[0,1],'--k')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.ylabel('Empirical Cummulative Distribution')
        plt.xlabel('Theoretical Cummulative Distribution')
        plt.savefig('%s/pp_plot_%04d.png' % (outdir,i_epoch),dpi=360)
        plt.savefig('%s/latest/latest_pp_plot.png' % outdir,dpi=360)
        plt.close()
        return

    def plot_y_test(self,i_epoch,x_test,y_test,sig_test,y_normscale):
        """
        Plot examples of test y-data generation
        """
        params = self.params

        # generate test data
        #x_test, y_test, x, sig_test, parnames = data_maker.generate(
        #    tot_dataset_size=params['n_samples'],
        #    ndata=params['ndata'],
        #    usepars=params['usepars'],
        #    sigma=params['sigma'],
        #    seed=params['seed']
        #)

        fig, axes = plt.subplots(params['r'],params['r'],figsize=(6,6),sharex='col',sharey='row')
        fig_zoom, axes_zoom = plt.subplots(params['r'],params['r'],figsize=(6,6),sharex='col',sharey='row')

        # run the x test data through the model
        x = x_test[:params['r']*params['r'],:]
        y_test = y_test[:params['r']*params['r'],:]
        sig_test = sig_test[:params['r']*params['r'],:]

        # apply forward model to the x data
        _,_,y = VICI_forward_model.run(params, x, y_test, np.shape(y_test)[1], "forward_model_dir_%s/forward_model.ckpt" % params['run_label'])

        y*=y_normscale[0]
        y_test *= y_normscale[0]

        cnt = 0
        for i in range(params['r']):
            for j in range(params['r']):

                axes[i,j].clear()
                axes[i,j].plot(np.arange(params['ndata'])/float(params['ndata']),y[cnt,:],'b-')
                axes[i,j].plot(np.arange(params['ndata'])/float(params['ndata']),y_test[cnt,:],'k',alpha=0.5)
                axes[i,j].plot(np.arange(params['ndata'])/float(params['ndata']),sig_test[cnt,:],'cyan',alpha=0.5)
                axes[i,j].set_xlim([0,1])
                axes[i,j].set_xlabel('t') if i==params['r']-1 else axes[i,j].set_xlabel('')
                axes[i,j].set_ylabel('y') if j==0 else axes[i,j].set_ylabel('')
                if i==0 and j==0:
                    axes[i,j].legend(('pred y','y'))

                axes_zoom[i,j].clear()
                axes_zoom[i,j].plot(np.arange(params['ndata'])/float(params['ndata']),y[cnt,:],'b-')
                axes_zoom[i,j].plot(np.arange(params['ndata'])/float(params['ndata']),y_test[cnt,:],'k',alpha=0.5)
                axes_zoom[i,j].plot(np.arange(params['ndata'])/float(params['ndata']),sig_test[cnt,:],'cyan',alpha=0.5)
                axes_zoom[i,j].set_xlim([0.4,0.6])
                axes_zoom[i,j].set_xlabel('t') if i==params['r']-1 else axes_zoom[i,j].set_xlabel('')
                axes_zoom[i,j].set_ylabel('y') if j==0 else axes_zoom[i,j].set_ylabel('')
                if i==0 and j==0:
                    axes_zoom[i,j].legend(('pred y','y'))
                cnt += 1

        fig.canvas.draw()
        fig.savefig('%s/ytest_%04d.png' % (params['plot_dir'][0],i_epoch),dpi=360)
        fig.savefig('%s/latest/latest_ytest.png' % params['plot_dir'],dpi=360)
        plt.close(fig)

        fig_zoom.canvas.draw()
        fig_zoom.savefig('%s/ytestZoom_%04d.png' % (params['plot_dir'][0],i_epoch),dpi=360)
        fig_zoom.savefig('%s/latest/latest_ytestZoom.png' % params['plot_dir'],dpi=360)
        plt.close(fig_zoom)
        return

    def make_loss_plot(self,KL_PLOT,COST_PLOT,i):
        # make log loss plot
        fig_loss, axes_loss = plt.subplots(1,figsize=(10,8))
        axes_loss.grid()
        axes_loss.set_ylabel('Loss')
        axes_loss.set_xlabel('Iterations elapsed: %s' % i)
        axes_loss.semilogx(np.arange(len(KL_PLOT)), KL_PLOT, label='KL')
        axes_loss.semilogx(np.arange(len(COST_PLOT)), COST_PLOT, label='COST')
        axes_loss.semilogx(np.arange(len(COST_PLOT)), (KL_PLOT+COST_PLOT), label='Total')
        axes_loss.legend(loc='upper left')
        plt.savefig('%s/latest/losses_logscale.png' % self.params['plot_dir'])
        plt.close(fig_loss)

        # make non-log scale loss plot
        fig_loss, axes_loss = plt.subplots(1,figsize=(10,8))
        axes_loss.grid()
        axes_loss.set_ylabel('Loss')
        axes_loss.set_xlabel('Iterations elapsed: %s' % i)
        axes_loss.plot(np.arange(len(KL_PLOT)), KL_PLOT, label='KL')
        axes_loss.plot(np.arange(len(COST_PLOT)), COST_PLOT, label='COST')
        axes_loss.plot(np.arange(len(COST_PLOT)), (KL_PLOT+COST_PLOT), label='Total')
        axes_loss.set_xscale('log')
        axes_loss.set_yscale('log')
        axes_loss.legend(loc='upper left')
        plt.savefig('%s/latest/losses.png' % self.params['plot_dir'])
        plt.close(fig_loss)


    def gen_kl_plots(self,model,sig_test,par_test,normscales):


        """
        Make kl corner histogram plots. Currently writing such that we 
        still bootstrap a split between samplers with themselves, but 
        will rewrite that once I find a way to run condor on 
        Bilby sampler runs.
        """

        def load_test_set(model,sig_test,par_test,normscales,sampler='dynesty1'):
            """
            load requested test set
            """

            if sampler=='vitamin1' or sampler=='vitamin2':
                # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                _, _, x, _ = model.run(self.params, sig_test, np.shape(par_test)[1], "inverse_model_dir_%s/inverse_model.ckpt" % self.params['run_label'])

                # Convert XS back to unnormalized version
                if self.params['do_normscale']:
                    for m in range(self.params['ndim_x']):
                        x[:,m,:] = x[:,m,:]*normscales[m] 
                return x               

            # Define variables
            pos_test = []
            samples = np.zeros((params['r']*params['r'],params['n_samples'],params['ndim_x']+1))
            cnt=0
            test_set_dir = params['kl_set_dir'] + '_' + sampler

            # Load test set
            for i in range(params['r']):
                for j in range(params['r']):
                    # TODO: remove this bandaged phase file calc
                    f = h5py.File('%s/test_samp_%d.h5py' % (test_set_dir,cnt), 'r+')

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
                    cnt += 1
                    f.close()

            pos_test = np.array(pos_test)
            
            pos_test = pos_test[:,[0,2,3,4]]
            samples = samples[:,:,[0,2,3,4]]
            samples = samples.reshape(samples.shape[0],samples.shape[2],samples.shape[1])

            return samples

        def compute_kl(sampset_1,sampset_2):
            """
            Compute KL for one test case.
            """
            # Remove samples outside of the prior mass distribution           
            cur_max = self.params['n_samples']
            set1 = []
            set2 = []

            # Iterate over parameters and remove samples outside of prior
            for i in range(sampset_1.shape[0]):
                mask = [(sampset_1[0,:] >= sampset_1[2,:]) & (sampset_1[3,:] >= 1000.0) & (sampset_1[3,:] <= 3000.0) & (sampset_1[1,:] >= 0.4) & (sampset_1[1,:] <= 0.6) & (sampset_1[0,:] >= 35.0) & (sampset_1[0,:] <= 80.0) & (sampset_1[2,:] <= 80.0) & (sampset_1[2,:] >= 35.0)]
                mask = np.argwhere(mask[0])
                new_rev = sampset_1[i,mask]
                new_rev = new_rev.reshape(new_rev.shape[0])
                new_samples = sampset_2[i,mask]
                new_samples = new_samples.reshape(new_samples.shape[0])
                tmp_max = new_rev.shape[0]
                if tmp_max < cur_max: cur_max = tmp_max
                set1.append(new_rev[:cur_max])
                set2.append(new_samples[:cur_max])

            set1 = np.array(set1)
            set2 = np.array(set2)

            kl_samps = []
            n_samps = self.params['n_samples']
            n_pars = self.params['ndim_x']

            # Iterate over number of randomized sample slices
            for j in range(self.params['n_kl_samp']):
                kl_result = np.sum(set2[:,np.random.randint(0,high=int(set2.shape[1]),size=int(set2.shape[1]/2.0))] * ( np.log(set2[:,np.random.randint(0,high=int(set2.shape[1]),size=int(set2.shape[1]/2.0))])
                                            - np.log(set1[:,np.random.randint(0,high=int(set2.shape[1]),size=int(set2.shape[1]/2.0))]) ))
                kl_samps.append(kl_result)
            kl_arr = kl_samps   

            return kl_arr
   
        # Define variables 
        params = self.params
        usesamps = params['use_samplers']
        samplers = params['samplers']
        fig_kl, axis_kl = plt.subplots(len(usesamps),len(usesamps),figsize=(6,6))
        
        # Compute kl divergence on all test cases with preds vs. benchmark
        # Iterate over samplers
        tmp_idx=len(usesamps)
        for i in range(len(usesamps)):
            for j in range(tmp_idx):
                # Load appropriate test sets
                if samplers[usesamps[i]] == samplers[usesamps[::-1][j]]:
                    set1 = load_test_set(model,sig_test,par_test,normscales,sampler=samplers[usesamps[i]]+'1')
                    set2 = load_test_set(model,sig_test,par_test,normscales,sampler=samplers[usesamps[::-1][j]]+'2')
                else:
                    set1 = load_test_set(model,sig_test,par_test,normscales,sampler=samplers[usesamps[i]]+'1')
                    set2 = load_test_set(model,sig_test,par_test,normscales,sampler=samplers[usesamps[::-1][j]]+'1')

                # Iterate over test cases
                tot_kl = []
                for r in range(self.params['r']**2):
                    tot_kl.append(compute_kl(set1[r],set2[r]))
                tot_kl = np.array(tot_kl)
                tot_kl = tot_kl.flatten()

                # Plot KL results
                axis_kl[len(usesamps)-1-j,i].hist(tot_kl,bins=100,alpha=0.5,color='blue',normed=True)
                for xtick,ytick in zip(axis_kl[i,j].xaxis.get_major_ticks(),axis_kl[i,j].yaxis.get_major_ticks()):
                    xtick.label.set_fontsize(4)
                    ytick.label.set_fontsize(4)

                #axis_kl[i,j].set_xlabel('KL Values') if i==self.params['r']-1 else axis_kl[i,j].set_xlabel('')

            tmp_idx -= 1 

        # Save KL corner plot
        fig_kl.canvas.draw()
        fig_kl.savefig('%s/latest/hist-kl.png' % (self.params['plot_dir'][0]),dpi=360)
        plt.close(fig_kl)

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
                    fig, axes = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),sharex='all',sharey='all')

                    # initialize 1D plots for showing testing results
                    fig_1d, axes_1d = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),sharex='all',sharey='all')

                    # initialize 1D plots for showing testing results for last 1d hist
                    fig_1d_last, axes_1d_last = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),sharex='all',sharey='all')

                    # initialize 1D plots for showing testing results
                    #fig_kl, axis_kl = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6))
                    fig_ad, axis_ad = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6))
                    fig_ks, axis_ks = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6))

                    # initialize 1D plots for showing testing results for last 1d hist
                    #fig_kl_last, axis_kl_last = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6))
                    fig_ad_last, axis_ad_last = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6))
                    fig_ks_last, axis_ks_last = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6))

                    # Iterate over test cases
                    for i in range(self.params['r']):
                        for j in range(self.params['r']):

                            # remove samples outside of the prior mass distribution
                            mask = [(self.rev_x[cnt,0,:] >= self.rev_x[cnt,2,:]) & (self.rev_x[cnt,3,:] >= 1000.0) & (self.rev_x[cnt,3,:] <= 3000.0) & (self.rev_x[cnt,1,:] >= 0.4) & (self.rev_x[cnt,1,:] <= 0.6) & (self.rev_x[cnt,0,:] >= 35.0) & (self.rev_x[cnt,0,:] <= 80.0) & (self.rev_x[cnt,2,:] <= 80.0) & (self.rev_x[cnt,2,:] >= 35.0)]
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
                            axes[i,j].set_xlabel(parname1) if i==self.params['r']-1 else axes[i,j].set_xlabel('')
                            axes[i,j].set_ylabel(parname2) if j==0 else axes[i,j].set_ylabel('')
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
                            n_hist_bins=25
                            axes_1d[i,j].clear()
                            axes_1d[i,j].hist(self.samples[cnt,mask,k].reshape(mask.shape[0]),color='b',bins=n_hist_bins,alpha=0.5,normed=True)
                            axes_1d[i,j].hist(self.rev_x[cnt,k,mask].reshape(mask.shape[0]),color='r',bins=n_hist_bins,alpha=0.5,normed=True)
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
                            matplotlib.rc('xtick', labelsize=8)
                            matplotlib.rc('ytick', labelsize=8)
                            axes_1d[i,j].set_xlabel(parname1) if i==self.params['r']-1 else axes_1d[i,j].set_xlabel('')

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
                                axis_ks[i,j].set_xlabel(parname1) if i==self.params['r']-1 else axis_ks[i,j].set_xlabel('')
                                axis_ad[i,j].set_xlabel(parname1) if i==self.params['r']-1 else axis_ad[i,j].set_xlabel('')

                            except IndexError:
                                print('Warning: bad stat result!')
                                continue

                            if i == 0 and j == 0: 
                                axis_kl[i,j].legend(loc='upper left', fontsize='x-small')
                                axis_ad[i,j].legend(loc='upper left', fontsize='x-small')
                                axis_ks[i,j].legend(loc='upper left', fontsize='x-small')

                            if k == (self.params['ndim_x']-2):
                                # plot the 1D samples and the 5% confidence bounds
                                ol_hist = data_maker.overlap(self.samples[cnt,mask,k+1].reshape(mask.shape[0],1),self.rev_x[cnt,k+1,mask].reshape(mask.shape[0],1),k)
                                olvec_1d[i,j,k+1] = ol_hist
                                axes_1d_last[i,j].clear()
                                axes_1d_last[i,j].hist(self.samples[cnt,mask,k+1].reshape(mask.shape[0]),color='b',bins=n_hist_bins,alpha=0.5,normed=True)
                                axes_1d_last[i,j].hist(self.rev_x[cnt,k+1,mask].reshape(mask.shape[0]),color='r',bins=n_hist_bins,alpha=0.5,normed=True)
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
                                        axis_ks[i,j].set_xlabel(parname1) if i==self.params['r']-1 else axis_ks[i,j].set_xlabel('')
                                        axis_ad[i,j].set_xlabel(parname1) if i==self.params['r']-1 else axis_ad[i,j].set_xlabel('')
                                    except IndexError:
                                        print('Warning: bad stat result!')
                                        continue

                            cnt += 1

                        # save the results to file
                        fig_1d.canvas.draw()
                        fig_1d.savefig('%s/latest/latest-1d_%d.png' % (self.params['plot_dir'][0],k),dpi=360)

                        if self.params['do_adkskl_test']:
                            fig_kl.canvas.draw()
                            fig_kl.savefig('%s/latest/hist-kl_%d.png' % (self.params['plot_dir'][0],k),dpi=360)
                            fig_ad.canvas.draw()
                            fig_ad.savefig('%s/latest/hist-ad_%d.png' % (self.params['plot_dir'][0],k),dpi=360)
                            fig_ks.canvas.draw()
                            fig_ks.savefig('%s/latest/hist-ks_%d.png' % (self.params['plot_dir'][0],k),dpi=360)
                            plt.close(fig_kl)
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
                                fig_kl_last.canvas.draw()
                                fig_kl_last.savefig('%s/latest/hist-kl_%d.png' % (self.params['plot_dir'][0],k+1),dpi=360)
                                plt.close(fig_kl_last)
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
