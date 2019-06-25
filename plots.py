import os, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np
from scipy.stats import uniform, norm, gaussian_kde, ks_2samp, anderson_ksamp
from scipy import stats

from data import chris_data as data_maker
from Models import VICI_forward_model
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

    def plot_y_test(self,i_epoch):
        """
        Plot examples of test y-data generation
        """
        params = self.params

        # generate test data
        x_test, y_test, x, sig_test, parnames = data_maker.generate(
            tot_dataset_size=params['n_samples'],
            ndata=params['ndata'],
            usepars=params['usepars'],
            sigma=params['sigma'],
            seed=params['seed']
        )

        fig, axes = plt.subplots(params['r'],params['r'],figsize=(6,6),sharex='col',sharey='row')

        # run the x test data through the model
        x = x_test[:params['r']*params['r'],:]
        y_test = y_test[:params['r']*params['r'],:]
        sig_test = sig_test[:params['r']*params['r'],:]

        # apply forward model to the x data
        _,_,y = VICI_forward_model.run(params, x, y_test, np.shape(y_test)[1], "forward_model_dir/forward_model.ckpt")

        cnt = 0
        for i in range(params['r']):
            for j in range(params['r']):

                axes[i,j].clear()
                axes[i,j].plot(np.arange(params['ndata'])/float(params['ndata']),y[cnt,:],'b-')
                axes[i,j].plot(np.arange(params['ndata'])/float(params['ndata']),y_test[cnt,:],'k',alpha=0.5)
                axes[i,j].set_xlim([0,1])
                #matplotlib.rc('xtick', labelsize=5)
                #matplotlib.rc('ytick', labelsize=5)
                axes[i,j].set_xlabel('t') if i==params['r']-1 else axes[i,j].set_xlabel('')
                axes[i,j].set_ylabel('y') if j==0 else axes[i,j].set_ylabel('')
                if i==0 and j==0:
                    axes[i,j].legend(('pred y','y'))
                cnt += 1

        fig.canvas.draw()
        plt.savefig('%s/ytest_%04d.png' % (params['plot_dir'][0],i_epoch),dpi=360)
        fig.savefig('%s/latest/latest_ytest.png' % params['plot_dir'],dpi=360)
        plt.close(fig)
        return

    def plot_y_dist(self,i_epoch):
        """
        Plots the joint distributions of y variables
        """
        params=self.params
        Nsamp = 1000

        # generate test data
        x_test, y_test, x, sig_test, parnames = data_maker.generate(
            tot_dataset_size=Nsamp,
            ndata=params['ndata'],
            usepars=params['usepars'],
            sigma=params['sigma'],
            seed=params['seed']
        )

        # run the x test data through the model
        x = x_test
        y_test = y_test
        sig_test = sig_test

        # apply forward model to the x data
        _,_,y = VICI_forward_model.run(params, x, y_test, np.shape(y_test)[1], "forward_model_dir/forward_model.ckpt")

        sig_test = sig_test
        dy = y - y_test
        """
        C = np.cov(dy.transpose())

        fig, axes = plt.subplots(1,figsize=(5,5))

        im = axes.imshow(C)

        # We want to show all ticks...
        axes.set_xticks(np.arange(params['ndata']))
        axes.set_yticks(np.arange(params['ndata']))

        # Rotate the tick labels and set their alignment.
        plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(params['ndata']):
            for j in range(params['ndata']):
                text = axes.text(j,i,'%.2f' % C[i,j], fontsize=3,
                           ha="center",va="center",color="w")

        fig.tight_layout()
        plt.savefig('%s/latest/latest_cov_y.png' % params['plot_dir'],dpi=360)
        plt.close(fig)
        """

        fig, axes = plt.subplots(1,figsize=(5,5))
        delta = np.transpose(y[:,:]-y_test[:,:])
        dyvec = np.linspace(-10*params['sigma'],10*params['sigma'],250)
        for d in delta:
            plt.hist(np.array(d).flatten(),25,density=True,histtype='stepfilled',alpha=0.5)
        plt.hist(np.array(delta).flatten(),25,density=True,histtype='step',linestyle='dashed')
        plt.plot(dyvec,norm.pdf(dyvec,loc=0,scale=np.sqrt(2.0)*params['sigma']),'k-')
        plt.xlabel('y-y_pred')
        plt.ylabel('p(y-y_pred)')
        plt.savefig('%s/y_dist_%04d.png' % (params['plot_dir'][0],i_epoch),dpi=360)
        plt.savefig('%s/latest/y_dist.png' % params['plot_dir'],dpi=360)
        plt.close(fig)
        return

    def plot_x_evolution(self):
        """
        Plot examples of test y-data generation
        """
        Nsamp = 100
        out_shape = [-1,ndim_tot]
        if conv==True:
            in_shape = [-1,1,ndim_tot]
        else:
            in_shape = [-1,ndim_tot]
        fig, axes = plt.subplots(ndim_x,ndim_x,figsize=(6,6))

        # make a noisy signal in the middle of the space
        t = np.arange(ndim_y)/float(ndim_y)
        A,t0,tau,p,w = np.array([0.5,0.5,0.5,0.2,0.2])
        fnyq = 0.5*len(t)
        s = A*np.sin(2.0*np.pi*(w*fnyq*(t-t0) + p))*np.exp(-((t-t0)/tau)**2)

        y_orig = s + np.random.normal(loc=0.0,scale=sigma,size=ndim_y)
        y = torch.tensor(np.tile(np.array(y_orig),Nsamp+1).reshape(Nsamp+1,ndim_y),dtype=torch.float,device=dev)

        # make random colors
        cols = ['r','b','g']

        # loop over different shells of z
        for j in range(3):

            # make specific z values
            temp = np.random.normal(loc=0.0,scale=1.0,size=(Nsamp+1,ndim_z))
            z = (j+1)*np.array([t/np.linalg.norm(t) for t in temp])
            z = torch.tensor(z,dtype=torch.float,device=dev)
            pad_yz = torch.zeros(Nsamp+1,ndim_tot-ndim_y-ndim_z,device=dev)
            yz_padded = torch.cat((y,z,pad_yz),dim=1)

            # apply backward model to the padded yz data
            if do_double_nn:
                if do_cnn:
                    data = torch.cat((y,z), dim=1)
                    output = model_r(data.reshape(data.shape[0],1,data.shape[1]))#.reshape(out_shape)
                    output_x = output[:,:ndim_x]  # extract the model output y
                else:
                    output = model_r(torch.cat((y,z), dim=1))#.reshape(out_shape)
                    output_x = output[:,:ndim_x]  # extract the model output y
            else:
                output = model(yz_padded.reshape(in_shape),rev=True)#.reshape(out_shape)
                output_x = output[:,model.inSchema.amp[0]:model.inSchema.tau[-1]+1]  # extract the model output y
            x = output_x.cpu().data.numpy()

            # loop over input parameters
            for i in range(ndim_x):
                for k in range(ndim_x):
                    if k<i:
                        axes[i,k].plot(x[:,i],x[:,k],'.',markersize=0.5,color=cols[j])
                        axes[i,k].set_xlim([0,1])
                        axes[i,k].set_ylim([0,1])
                        matplotlib.rc('xtick', labelsize=8)
                        matplotlib.rc('ytick', labelsize=8)
                        axes[i,k].set_xlabel(parnames[i])
                        axes[i,k].set_ylabel(parnames[k])
                    elif k==ndim_x-2 and i==ndim_x-2:
                        axes[i,k].plot(np.arange(ndim_y)/float(ndim_y),y_orig,'b-')
                        axes[i,k].plot(np.arange(ndim_y)/float(ndim_y),s,'r-')
                        axes[i,k].set_xlim([0,1])
                        axes[i,k].set_ylim([-1,1])
                        matplotlib.rc('xtick', labelsize=8)
                        matplotlib.rc('ytick', labelsize=8)
                        axes[i,k].set_xlabel('t')
                        axes[i,k].set_ylabel('y')
                    else:
                        axes[i,k].axis('off')

        fig.canvas.draw()
        plt.savefig('%s/xevo_%04d.png' % (outdir,i_epoch),dpi=360)
        plt.savefig('%s/latest/latest_xevo.png' % (outdir),dpi=360)
        plt.close()
        return

    def plot_z_dist(self):
        """
        Plots the distribution of latent z variables
        """
        Nsamp = 250
        out_shape = [-1,ndim_tot]
        if conv==True:
            in_shape = [-1,1,ndim_tot]
        else:
            in_shape = [-1,ndim_tot]

        # generate test data
        x_test, y_test, x, sig_test, parnames = data_maker.generate(
            tot_dataset_size=Nsamp,
            ndata=ndim_y,
            usepars=usepars,
            sigma=sigma,
            seed=1
        )

        # run the x test data through the model
        x = torch.tensor(x_test,dtype=torch.float,device=dev).clone().detach()
        y_test = torch.tensor(y_test,dtype=torch.float,device=dev).clone().detach()
        sig_test = torch.tensor(sig_test,dtype=torch.float,device=dev).clone().detach()

        # make the new padding for the noisy data and latent vector data
        pad_x = torch.zeros(Nsamp,ndim_tot-ndim_x-ndim_y,device=dev)

        # make a padded zy vector (with all new noise)
        x_padded = torch.cat((x,pad_x,y_test-sig_test),dim=1)

        # apply forward model to the x data
        if do_double_nn:
            if do_cnn:
                data = torch.cat((x,y_test-sig_test), dim=1)
                output = model_f(data.reshape(data.shape[0],1,data.shape[1]))#.reshape(out_shape)
                output_z = output[:,ndim_y:]  # extract the model output y
            else:
                output = model_f(torch.cat((x,y_test-sig_test), dim=1))#.reshape(out_shape)
                output_z = output[:,ndim_y:]  # extract the model output y
        else:
            output = model(x_padded.reshape(in_shape))#.reshape(out_shape)
            output_z = output[:,model.outSchema.LatentSpace]  # extract the model output y
        z = output_z.cpu().data.numpy()
        C = np.cov(z.transpose())

        fig, axes = plt.subplots(1,figsize=(5,5))
        im = axes.imshow(np.abs(C))

        # We want to show all ticks...
        axes.set_xticks(np.arange(ndim_z))
        axes.set_yticks(np.arange(ndim_z))

        # Rotate the tick labels and set their alignment.
        plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(ndim_z):
            for j in range(ndim_z):
                text = axes.text(j,i,'%.2f' % C[i,j], fontsize=3,
                           ha="center",va="center",color="w")

        fig.tight_layout()
        fig.savefig('%s/cov_z_%04d.png' % (outdir,i_epoch),dpi=360)
        fig.savefig('%s/latest/latest_cov_z.png' % outdir,dpi=360)
        plt.close(fig)

        fig, axes = plt.subplots(ndim_z,ndim_z,figsize=(5,5))
        for c in range(ndim_z):
            for d in range(ndim_z):
                if d<c:
                    patches = []
                    axes[c,d].clear()
                    matplotlib.rc('xtick', labelsize=8)
                    matplotlib.rc('ytick', labelsize=8)
                    axes[c,d].plot(z[:,c],z[:,d],'.r',markersize=0.5)
                    circle1 = Circle((0.0, 0.0), 1.0,fill=False,linestyle='--')
                    patches.append(circle1)
                    circle2 = Circle((0.0, 0.0), 2.0,fill=False,linestyle='--')
                    patches.append(circle2)
                    circle3 = Circle((0.0, 0.0), 3.0,fill=False,linestyle='--')
                    patches.append(circle3)
                    p = PatchCollection(patches, alpha=0.2)
                    axes[c,d].add_collection(p)
                    axes[c,d].set_yticklabels([])
                    axes[c,d].set_xticklabels([])
                    axes[c,d].set_xlim([-3,3])
                    axes[c,d].set_ylim([-3,3])
                else:
                    axes[c,d].axis('off')
                axes[c,d].set_xlabel('')
                axes[c,d].set_ylabel('')

        fig.savefig('%s/scatter_z_%04d.png' % (outdir,i_epoch),dpi=360)
        fig.savefig('%s/latest/latest_scatter_z.png' % outdir,dpi=360)
        plt.close(fig)

        fig, axes = plt.subplots(1,figsize=(5,5))
        delta = np.transpose(z[:,:])
        dyvec = np.linspace(-10*1.0,10*1.0,250)
        for d in delta:
            plt.hist(np.array(d).flatten(),25,density=True,histtype='stepfilled',alpha=0.5)
        plt.hist(np.array(delta).flatten(),25,density=True,histtype='step',linestyle='dashed')
        plt.plot(dyvec,norm.pdf(dyvec,loc=0,scale=1.0),'k-')
        plt.xlabel('predicted z')
        plt.ylabel('p(z)')

        fig.savefig('%s/dist_z_%04d.png' % (outdir,i_epoch),dpi=360)
        fig.savefig('%s/latest/latest_dist_z.png' % outdir,dpi=360)
        plt.close(fig)

        return


    # Plot predicted time series vs. actually time series examples
    """
    # Make x evolution plot
    plot_x_evolution(model,ndim_x,ndim_y,ndim_z,ndim_tot,sigma,parnames,out_dir,epoch,conv=False)
    # Make z distribution plots
    plot_z_dist(model,ndim_x,ndim_y,ndim_z,ndim_tot,usepars,sigma,out_dir,epoch,conv=False)
    """

    def make_overlap_plot(self):
        olvec = np.zeros((self.params['r'],self.params['r'],1))
        olvec_1d = np.zeros((self.params['r'],self.params['r'],self.params['ndim_x']))

        # Make 2D scatter plots of posteriors
        for k in range(self.params['ndim_x']):
            parname1 = self.params['parnames'][k]
            for nextk in range(self.params['ndim_x']):
                parname2 = self.params['parnames'][nextk]
                if nextk>k:
                    cnt = 0

                    # initialize 2D plot for showing testing results
                    fig, axes = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),sharex='col',sharey='row')

                    # initialize 1D plots for showing testing results
                    fig_1d, axes_1d = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),sharex='col',sharey='row')

                    # initialize 1D plots for showing testing results for last 1d hist
                    fig_1d_last, axes_1d_last = plt.subplots(self.params['r'],self.params['r'],figsize=(6,6),sharex='col',sharey='row')

                    # Iterate over test cases
                    for i in range(self.params['r']):
                        for j in range(self.params['r']):
                            # compute the n-d overlap
                            if k==0 and nextk==1:
                                ol = data_maker.overlap(self.samples[cnt,:,:],self.rev_x[cnt,:,:].T)
                                olvec[i,j,0] = ol

                            # plot the samples and the true contours
                            axes[i,j].clear()
                            axes[i,j].scatter(self.samples[cnt,:,k], self.samples[cnt,:,nextk],c='b',s=0.2,alpha=0.5, label='MCMC')
                            axes[i,j].scatter(self.rev_x[cnt,k,:], self.rev_x[cnt,nextk,:],c='r',s=0.2,alpha=0.5, label='INN')
                            axes[i,j].set_xlim([0,1])
                            axes[i,j].set_ylim([0,1])
                            axes[i,j].plot(self.pos_test[cnt,k],self.pos_test[cnt,nextk],'+c',markersize=8, label='Truth')
                            oltxt = '%.2f' % olvec[i,j,0]
                            axes[i,j].text(0.90, 0.95, oltxt,
                                horizontalalignment='right',
                                verticalalignment='top',
                                    transform=axes[i,j].transAxes)
                            matplotlib.rc('xtick', labelsize=8)
                            matplotlib.rc('ytick', labelsize=8)
                            axes[i,j].set_xlabel(parname1) if i==self.params['r']-1 else axes[i,j].set_xlabel('')
                            axes[i,j].set_ylabel(parname2) if j==0 else axes[i,j].set_ylabel('')
                            if i == 0 and j == 0: axes[i,j].legend(loc='upper left', fontsize='x-small')
                            #cnt += 1

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
                            ol_hist = data_maker.overlap(self.samples[cnt,:,k].reshape(self.params['n_samples'],1),self.rev_x[cnt,k,:].reshape(self.params['n_samples'],1))
                            olvec_1d[i,j,k] = ol_hist
                            n_hist_bins=30
                            axes_1d[i,j].clear()
                            axes_1d[i,j].hist(self.samples[cnt,:,k],color='b',bins=n_hist_bins,range=(0,1),alpha=0.5,normed=True)
                            axes_1d[i,j].hist(self.rev_x[cnt,k,:],color='r',bins=n_hist_bins,range=(0,1),alpha=0.5,normed=True)
                            axes_1d[i,j].set_xlim([0,1])
                            axes_1d[i,j].axvline(x=self.pos_test[cnt,k], linewidth=0.5, color='black')
                            axes_1d[i,j].axvline(x=confidence_bd(self.samples[cnt,:,k])[0], linewidth=0.5, color='b')
                            axes_1d[i,j].axvline(x=confidence_bd(self.samples[cnt,:,k])[1], linewidth=0.5, color='b')
                            axes_1d[i,j].axvline(x=confidence_bd(self.rev_x[cnt,k,:])[0], linewidth=0.5, color='r')
                            axes_1d[i,j].axvline(x=confidence_bd(self.rev_x[cnt,k,:])[1], linewidth=0.5, color='r')
                            oltxt = '%.2f' % olvec_1d[i,j,k]
                            axes_1d[i,j].text(0.90, 0.95, oltxt,
                                horizontalalignment='right',
                                verticalalignment='top',
                                    transform=axes_1d[i,j].transAxes)
                            matplotlib.rc('xtick', labelsize=8)
                            matplotlib.rc('ytick', labelsize=8)
                            axes_1d[i,j].set_xlabel(parname1) if i==self.params['r']-1 else axes_1d[i,j].set_xlabel('')

                            if k == (self.params['ndim_x']-2):
                                # plot the 1D samples and the 5% confidence bounds
                                ol_hist = data_maker.overlap(self.samples[cnt,:,k+1].reshape(self.params['n_samples'],1),self.rev_x[cnt,k+1,:].reshape(self.params['n_samples'],1))
                                olvec_1d[i,j,k+1] = ol_hist
                                axes_1d_last[i,j].clear()
                                axes_1d_last[i,j].hist(self.samples[cnt,:,k+1],color='b',bins=n_hist_bins,range=(0,1),alpha=0.5,normed=True)
                                axes_1d_last[i,j].hist(self.rev_x[cnt,k+1,:],color='r',bins=n_hist_bins,range=(0,1),alpha=0.5,normed=True)
                                axes_1d_last[i,j].set_xlim([0,1])
                                axes_1d_last[i,j].axvline(x=self.pos_test[cnt,k+1], linewidth=0.5, color='black')
                                axes_1d_last[i,j].axvline(x=confidence_bd(self.samples[cnt,:,k+1])[0], linewidth=0.5, color='b')
                                axes_1d_last[i,j].axvline(x=confidence_bd(self.samples[cnt,:,k+1])[1], linewidth=0.5, color='b')
                                axes_1d_last[i,j].axvline(x=confidence_bd(self.rev_x[cnt,k+1,:])[0], linewidth=0.5, color='r')
                                axes_1d_last[i,j].axvline(x=confidence_bd(self.rev_x[cnt,k+1,:])[1], linewidth=0.5, color='r')
                                oltxt = '%.2f' % olvec_1d[i,j,k+1]
                                axes_1d_last[i,j].text(0.90, 0.95, oltxt,
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                        transform=axes_1d_last[i,j].transAxes)
                                axes_1d_last[i,j].set_xlabel(self.params['parnames'][k+1]) if i==self.params['r']-1 else axes_1d_last[i,j].set_xlabel('')

                            cnt += 1
                        # save the results to file
                        fig_1d.canvas.draw()
                        fig_1d.savefig('%s/latest/latest-1d_%d.png' % (self.params['plot_dir'][0],k),dpi=360)

                        if k == (self.params['ndim_x']-2):
                            # save the results to file
                            fig_1d_last.canvas.draw()
                            fig_1d_last.savefig('%s/latest/latest-1d_%d.png' % (self.params['plot_dir'][0],k+1),dpi=360)

                        # save the results to file
                        fig.canvas.draw()
                        fig.savefig('%s/latest/posteriors_%d%d.png' % (self.params['plot_dir'][0],k,nextk),dpi=360)
                        plt.close(fig)

