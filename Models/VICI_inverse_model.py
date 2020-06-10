################################################################################################################
#
# --Variational Inference for Computational Imaging (VICI) Inverse Problem--
# 
# This model takes as input measured signals and infers target images/objects.
################################################################################################################

import numpy as np
import time
import os, sys,io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import corner
import bilby_pe
import h5py

from Neural_Networks import VICI_decoder
from Neural_Networks import VICI_encoder
from Neural_Networks import VICI_VAE_encoder
from Neural_Networks import batch_manager
from Models import VICI_inverse_model
import plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tfd = tfp.distributions
SMALL_CONSTANT = 1e-10 # necessary to prevent the division by zero in many operations 

def get_wrap_index(params):

    # identify the indices of wrapped and non-wrapped parameters - clunky code
    wrap_mask, nowrap_mask = [], []
    idx_wrap, idx_nowrap = [], []
    
    # loop over inference params
    for i,p in enumerate(params['inf_pars']):

        # loop over wrapped params 
        flag = False
        for q in params['wrap_pars']:
            if p==q:
                flag = True    # if inf params is a wrapped param set flag
        
        # record the true/false value for this inference param
        if flag==True:
            wrap_mask.append(True)
            nowrap_mask.append(False)
            idx_wrap.append(i)
        elif flag==False:
            wrap_mask.append(False)
            nowrap_mask.append(True)
            idx_nowrap.append(i)
     
    idx_mask = idx_nowrap + idx_wrap
    return wrap_mask, nowrap_mask, idx_mask

def load_chunk(input_dir,inf_pars,params,bounds,fixed_vals,load_condor=False):

    # load generated samples back in
    train_files = []
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': []}

    if load_condor == True:
        filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    else:
        filenames = os.listdir(dataLocations[0])

    snrs = []
    for filename in filenames:
        try:
            train_files.append(filename)
        except OSError:
            print('Could not load requested file')
            continue

    train_files_idx = np.arange(len(train_files))[:int(params['load_chunk_size']/1000.0)]
    np.random.shuffle(train_files)
    train_files = np.array(train_files)[train_files_idx]
    for filename in train_files: 
            print(filename)
            data_temp={'x_data': h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data'][:],
                  'y_data_noisefree': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisefree'][:],
                  'rand_pars': h5py.File(dataLocations[0]+'/'+filename, 'r')['rand_pars'][:]}
            data['x_data'].append(data_temp['x_data'])
            data['y_data_noisefree'].append(np.expand_dims(data_temp['y_data_noisefree'], axis=0))
            data['rand_pars'] = data_temp['rand_pars']


    # extract the prior bounds
    bounds = {}
    bounds = {}
    for k in data_temp['rand_pars']:
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'
        bounds[par_max] = h5py.File(dataLocations[0]+'/'+filename, 'r')[par_max][...].item()
        bounds[par_min] = h5py.File(dataLocations[0]+'/'+filename, 'r')[par_min][...].item()
    data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
    data['y_data_noisefree'] = np.concatenate(np.array(data['y_data_noisefree']), axis=0)


    # normalise the data parameters
    for i,k in enumerate(data_temp['rand_pars']):
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'

        data['x_data'][:,i]=(data['x_data'][:,i] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
    x_data = data['x_data']
    y_data = data['y_data_noisefree']

    # extract inference parameters
    idx = []
    for k in inf_pars:
        print(k)
        for i,q in enumerate(data['rand_pars']):
            m = q.decode('utf-8')
            if k==m:
                idx.append(i)
    x_data = x_data[:,idx]

    
    # reshape arrays for multi-detector
    y_data_train = y_data
    y_data_train = y_data_train.reshape(y_data_train.shape[0]*y_data_train.shape[1],y_data_train.shape[2]*y_data_train.shape[3])

    # reshape y data into channels last format for convolutional approach
    if params['n_filters_r1'] != None:
        y_data_train_copy = np.zeros((y_data_train.shape[0],params['ndata'],len(fixed_vals['det'])))

        for i in range(y_data_train.shape[0]):
            for j in range(len(fixed_vals['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_train_copy[i,:,j] = y_data_train[i,idx_range]
        y_data_train = y_data_train_copy

    return x_data, y_data_train

def train(params, x_data, y_data, x_data_test, y_data_test, y_data_test_noisefree, y_normscale, save_dir, truth_test, bounds, fixed_vals, posterior_truth_test,snrs_test=None):    
    """ Function for training the conditional variational autoencoder (CVAE)
    """

    # USEFUL SIZES
    xsh = np.shape(x_data)
   
    ysh = np.shape(y_data)[1]
    z_dimension = params['z_dimension']
    bs = params['batch_size']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']
    n_modes = params['n_modes']
    n_hlayers_r1 = len(params['n_weights_r1'])
    n_hlayers_r2 = len(params['n_weights_r2'])
    n_hlayers_q = len(params['n_weights_q'])
    n_conv_r1 = len(params['n_filters_r1'])
    n_conv_r2 = len(params['n_filters_r2'])
    n_conv_q = len(params['n_filters_q'])
    n_filters_r1 = params['n_filters_r1']
    n_filters_r2 = params['n_filters_r2']
    n_filters_q = params['n_filters_q']
    filter_size_r1 = params['filter_size_r1']
    filter_size_r2 = params['filter_size_r2']
    filter_size_q = params['filter_size_q']
    maxpool_r1 = params['maxpool_r1']
    maxpool_r2 = params['maxpool_r2']
    maxpool_q = params['maxpool_q']
    conv_strides_r1 = params['conv_strides_r1']
    conv_strides_r2 = params['conv_strides_r2']
    conv_strides_q = params['conv_strides_q']
    pool_strides_r1 = params['pool_strides_r1']
    pool_strides_r2 = params['pool_strides_r2']
    pool_strides_q = params['pool_strides_q']
    batch_norm = params['batch_norm']
    ysh_conv_r1 = int(ysh)
    ysh_conv_r2 = int(ysh)
    ysh_conv_q = int(ysh)
    drate = params['drate']
    ramp_start = params['ramp_start']
    ramp_end = params['ramp_end']
    num_det = len(fixed_vals['det'])


    # identify the indices of wrapped and non-wrapped parameters - clunky code
    wrap_mask, nowrap_mask, idx_mask = get_wrap_index(params)
    wrap_len = np.sum(wrap_mask)
    nowrap_len = np.sum(nowrap_mask)

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():

        # PLACE HOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_ph") # params placeholder
        if n_conv_r1 != None:
           if params['by_channel'] == True:
               y_ph = tf.placeholder(dtype=tf.float32, shape=[None,ysh,len(fixed_vals['det'])], name="y_ph")    # data placeholder
           else:
               y_ph = tf.placeholder(dtype=tf.float32, shape=[None,len(fixed_vals['det']),ysh], name="y_ph")
        else:
            y_ph = tf.placeholder(dtype=tf.float32, shape=[None,ysh], name="y_ph")    # data placeholder
        idx = tf.placeholder(tf.int32)

        # LOAD VICI NEURAL NETWORKS
        r2_xzy = VICI_decoder.VariationalAutoencoder('VICI_decoder', wrap_mask, nowrap_mask, 
                                                     n_input1=z_dimension, n_input2=ysh_conv_r2, n_output=xsh[1], 
                                                     n_weights=n_weights_r2, n_hlayers=n_hlayers_r2, 
                                                     drate=drate, n_filters=n_filters_r2, filter_size=filter_size_r2,
                                                     maxpool=maxpool_r2, n_conv=n_conv_r2, conv_strides=conv_strides_r2, pool_strides=pool_strides_r2, num_det=num_det, batch_norm=batch_norm, by_channel=params['by_channel'], weight_init=params['weight_init'])
        r1_zy = VICI_encoder.VariationalAutoencoder('VICI_encoder', n_input=ysh_conv_r1, n_output=z_dimension, 
                                                     n_weights=n_weights_r1, n_modes=n_modes, 
                                                     n_hlayers=n_hlayers_r1, drate=drate, n_filters=n_filters_r1, 
                                                     filter_size=filter_size_r1,maxpool=maxpool_r1, n_conv=n_conv_r1, conv_strides=conv_strides_r1, pool_strides=pool_strides_r1, num_det=num_det, batch_norm=batch_norm, by_channel=params['by_channel'], weight_init=params['weight_init'])
        q_zxy = VICI_VAE_encoder.VariationalAutoencoder('VICI_VAE_encoder', n_input1=xsh[1], n_input2=ysh_conv_q, 
                                                        n_output=z_dimension, n_weights=n_weights_q, 
                                                        n_hlayers=n_hlayers_q, drate=drate, n_filters=n_filters_q, 
                                                        filter_size=filter_size_q,maxpool=maxpool_q, n_conv=n_conv_q, conv_strides=conv_strides_q, pool_strides=pool_strides_q, num_det=num_det, batch_norm=batch_norm, by_channel=params['by_channel'], weight_init=params['weight_init']) # used to sample from q(z|x,y)?
        tf.set_random_seed(np.random.randint(0,10))

        ramp = (tf.log(tf.dtypes.cast(idx,dtype=tf.float32)) - tf.log(ramp_start))/(tf.log(ramp_end)-tf.log(ramp_start))
        ramp = tf.minimum(tf.math.maximum(0.0,ramp),1.0)
        
        if params['ramp'] == False:
            ramp = 1.0
 
        # reduce the y data size
        y_conv = y_ph

        # GET r1(z|y)
        # run inverse autoencoder to generate mean and logvar of z given y data - these are the parameters for r1(z|y)
        r1_loc, r1_scale, r1_weight = r1_zy._calc_z_mean_and_sigma(y_conv)
        r1_scale = tf.sqrt(SMALL_CONSTANT + tf.exp(r1_scale))
        # get l1 loss term
        l1_loss_weight = ramp*1e-3*tf.reduce_sum(tf.math.abs(r1_weight),1)
        r1_weight = ramp*tf.squeeze(r1_weight)
        
        # define the r1(z|y) mixture model
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=ramp*r1_weight),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_loc,
                          scale_diag=r1_scale))


        # DRAW FROM r1(z|y) - given the Gaussian parameters generate z samples
        r1_zy_samp = bimix_gauss.sample()        
        
        # GET q(z|x,y)
        q_zxy_mean, q_zxy_log_sig_sq = q_zxy._calc_z_mean_and_sigma(x_ph,y_conv)

        # DRAW FROM q(z|x,y)
        q_zxy_samp = q_zxy._sample_from_gaussian_dist(bs_ph, z_dimension, q_zxy_mean, tf.log(SMALL_CONSTANT + tf.exp(q_zxy_log_sig_sq)))
        
        # GET r2(x|z,y)
        reconstruction_xzy = r2_xzy.calc_reconstruction(q_zxy_samp,y_conv)
        r2_xzy_mean_nowrap = reconstruction_xzy[0]
        r2_xzy_log_sig_sq_nowrap = reconstruction_xzy[1]
        if np.sum(wrap_mask)>0:
            r2_xzy_mean_wrap = reconstruction_xzy[2]
            r2_xzy_log_sig_sq_wrap = reconstruction_xzy[3]

        # COST FROM RECONSTRUCTION - Gaussian parts
        normalising_factor_x = -0.5*tf.log(SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_nowrap)) - 0.5*np.log(2.0*np.pi)   # -0.5*log(sig^2) - 0.5*log(2*pi)
        square_diff_between_mu_and_x = tf.square(r2_xzy_mean_nowrap - tf.boolean_mask(x_ph,nowrap_mask,axis=1))         # (mu - x)^2

        inside_exp_x = -0.5 * tf.divide(square_diff_between_mu_and_x,SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_nowrap)) # -0.5*(mu - x)^2 / sig^2
        reconstr_loss_x = tf.reduce_sum(normalising_factor_x + inside_exp_x,axis=1,keepdims=True)                       # sum_dim(-0.5*log(sig^2) - 0.5*log(2*pi) - 0.5*(mu - x)^2 / sig^2)

        # COST FROM RECONSTRUCTION - Von Mises parts
        if np.sum(wrap_mask)>0:
            con = tf.reshape(tf.math.reciprocal(SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_wrap)),[-1,wrap_len])   # modelling wrapped scale output as log variance
            von_mises = tfp.distributions.VonMises(loc=2.0*np.pi*(tf.reshape(r2_xzy_mean_wrap,[-1,wrap_len])-0.5), concentration=con)   # define p_vm(2*pi*mu,con=1/sig^2)
            reconstr_loss_vm = tf.reduce_sum(von_mises.log_prob(2.0*np.pi*(tf.reshape(tf.boolean_mask(x_ph,wrap_mask,axis=1),[-1,wrap_len]) - 0.5)),axis=1)   # 2pi is the von mises input range
            cost_R = -1.0*tf.reduce_mean(reconstr_loss_x + reconstr_loss_vm) # average over batch
            r2_xzy_mean = tf.gather(tf.concat([r2_xzy_mean_nowrap,r2_xzy_mean_wrap],axis=1),tf.constant(idx_mask),axis=1)
            r2_xzy_scale = tf.gather(tf.concat([r2_xzy_log_sig_sq_nowrap,r2_xzy_log_sig_sq_wrap],axis=1),tf.constant(idx_mask),axis=1) 
        else:
            cost_R = -1.0*tf.reduce_mean(reconstr_loss_x)    
            r2_xzy_mean = r2_xzy_mean_nowrap
            r2_xzy_scale = r2_xzy_log_sig_sq_nowrap

        # compute montecarlo KL - first compute the analytic self entropy of q 
        normalising_factor_kl = -0.5*tf.log(SMALL_CONSTANT + tf.exp(q_zxy_log_sig_sq)) - 0.5*np.log(2.0*np.pi)   # -0.5*log(sig^2) - 0.5*log(2*pi)
        square_diff_between_qz_and_q = tf.square(q_zxy_mean - q_zxy_samp)                                        # (mu - x)^2
        inside_exp_q = -0.5 * tf.divide(square_diff_between_qz_and_q,SMALL_CONSTANT + tf.exp(q_zxy_log_sig_sq))  # -0.5*(mu - x)^2 / sig^2
        log_q_q = tf.reduce_sum(normalising_factor_kl + inside_exp_q,axis=1,keepdims=True)                       # sum_dim(-0.5*log(sig^2) - 0.5*log(2*pi) - 0.5*(mu - x)^2 / sig^2)
        log_r1_q = bimix_gauss.log_prob(q_zxy_samp)   # evaluate the log prob of r1 at the q samples
        KL = tf.reduce_mean(log_q_q - log_r1_q)      # average over batch

        # THE VICI COST FUNCTION
        COST = cost_R + ramp*KL

        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VICI")]
        
        # DEFINE OPTIMISER (using ADAM here)
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate']) 
        minimize = optimizer.minimize(COST,var_list = var_list_VICI)
        
        # INITIALISE AND RUN SESSION
        init = tf.global_variables_initializer()
        session.run(init)
        saver = tf.train.Saver()

    print('Training Inference Model...')    
    # START OPTIMISATION OF OELBO
    indices_generator = batch_manager.SequentialIndexer(params['batch_size'], xsh[0])
    plotdata = []
    if params['by_channel'] == False:
        y_data_test_new = []
        for sig in y_data_test:
            y_data_test_new.append(sig.T)
        y_data_test = np.array(y_data_test_new)
        del y_data_test_new


    load_chunk_it = 1
    for i in range(params['num_iterations']):

        next_indices = indices_generator.next_indices()

        # if load chunks true, load in data by chunks
        if params['load_by_chunks'] == True and i == int(params['load_iteration']*load_chunk_it):
            x_data, y_data = load_chunk(params['train_set_dir'],params['inf_pars'],params,bounds,fixed_vals)
            load_chunk_it += 1

        # Make noise realizations and add to training data
        next_x_data = x_data[next_indices,:]
        if n_conv_r1 != None:
            next_y_data = y_data[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],int(params['ndata']),len(fixed_vals['det'])))
        else:
            next_y_data = y_data[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],int(params['ndata']*len(fixed_vals['det']))))
        next_y_data /= y_normscale  # required for fast convergence

        if params['by_channel'] == False:
            next_y_data_new = [] 
            for sig in next_y_data:
                next_y_data_new.append(sig.T)
            next_y_data = np.array(next_y_data_new)
            del next_y_data_new
       
        # restore session if wanted
        if params['resume_training'] == True and i == 0 :
            print(save_dir)
            saver.restore(session, save_dir)
 
        # train to minimise the cost function
        session.run(minimize, feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, idx:i})

        # if we are in a report iteration extract cost function values
        if i % params['report_interval'] == 0 and i > 0:

            # get training loss
            cost, kl, AB_batch = session.run([cost_R, KL, r1_weight], feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, idx:i})

            # get validation loss on test set
            cost_val, kl_val = session.run([cost_R, KL], feed_dict={bs_ph:y_data_test.shape[0], x_ph:x_data_test, y_ph:y_data_test/y_normscale, idx:i})
            plotdata.append([cost,kl,cost+kl,cost_val,kl_val,cost_val+kl_val])

           
            try:
                # Make loss plot
                plt.figure()
                xvec = params['report_interval']*np.arange(np.array(plotdata).shape[0])
                plt.semilogx(xvec,np.array(plotdata)[:,0],label='recon',color='blue',alpha=0.5)
                plt.semilogx(xvec,np.array(plotdata)[:,1],label='KL',color='orange',alpha=0.5)
                plt.semilogx(xvec,np.array(plotdata)[:,2],label='total',color='green',alpha=0.5)
                plt.semilogx(xvec,np.array(plotdata)[:,3],label='recon_val',color='blue',linestyle='dotted')
                plt.semilogx(xvec,np.array(plotdata)[:,4],label='KL_val',color='orange',linestyle='dotted')
                plt.semilogx(xvec,np.array(plotdata)[:,5],label='total_val',color='green',linestyle='dotted')
                plt.ylim([-25,15])
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.legend()
                plt.savefig('%s/latest_%s/cost_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.ylim([np.min(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,0]), np.max(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,1])])
                plt.savefig('%s/latest_%s/cost_zoom_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.close('all')
                
            except:
                pass

            if params['print_values']==True:
                print('--------------------------------------------------------------')
                print('Iteration:',i)
                print('Training -ELBO:',cost)
                print('Validation -ELBO:',cost_val)
                print('Training KL Divergence:',kl)
                print('Validation KL Divergence:',kl_val)
                print('Training Total cost:',kl + cost) 
                print('Validation Total cost:',kl_val + cost_val)
                print()

                # terminate training if vanishing gradient
                if np.isnan(kl+cost) == True or np.isnan(kl_val+cost_val) == True:
                    print('Network is returning NaN values')
                    print('Terminating network training')
                    if params['hyperparam_optim'] == True:
                        save_path = saver.save(session,save_dir)
                        return 5000.0, session, saver, save_dir
                    else:
                        exit()
                try:
                    # Save loss plot data
                    np.savetxt(save_dir.split('/')[0] + '/loss_data.txt', np.array(plotdata))
                except FileNotFoundError as err:
                    print(err)
                    pass

        if i % params['save_interval'] == 0 and i > 0:

            if params['hyperparam_optim'] == False:
                # Save model 
                save_path = saver.save(session,save_dir)
            else:
                pass


        # stop hyperparam optim training it and return total loss as figure of merit
        if params['hyperparam_optim'] == True and i == params['hyperparam_optim_stop']:
            save_path = saver.save(session,save_dir)

            return np.array(plotdata)[-1,2], session, saver, save_dir

        if i % params['plot_interval'] == 0 and i>0:

            n_mode_weight_copy = 100 # must be a multiple of 50
            # just run the network on the test data
            for j in range(params['r']*params['r']):

                # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                if params['n_filters_r1'] != None:
                    XS, loc, scale, dt, _  = VICI_inverse_model.run(params, y_data_test[j].reshape([1,y_data_test.shape[1],y_data_test.shape[2]]), np.shape(x_data_test)[1],
                                                 y_normscale, 
                                                 "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
                else:
                    XS, loc, scale, dt, _  = VICI_inverse_model.run(params, y_data_test[j].reshape([1,-1]), np.shape(x_data_test)[1],
                                                 y_normscale, 
                                                 "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
                print('Runtime to generate {} samples = {} sec'.format(params['n_samples'],dt))            
               
                # Get corner parnames to use in plotting labels
                parnames = []
                for k_idx,k in enumerate(params['rand_pars']):
                    if np.isin(k, params['inf_pars']):
                        parnames.append(params['cornercorner_parnames'][k_idx])

                defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=[0.16, 0.84],
                    levels=(0.68,0.90,0.95), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

                figure = corner.corner(posterior_truth_test[j], **defaults_kwargs,labels=parnames,
                       color='tab:blue',truths=x_data_test[j,:],
                       show_titles=True)
                corner.corner(XS,**defaults_kwargs,labels=parnames,
                       color='tab:red',
                       fill_contours=True,
                       show_titles=True, fig=figure)


                plt.savefig('%s/corner_plot_%s_%d-%d.png' % (params['plot_dir'],params['run_label'],i,j))
                plt.savefig('%s/latest_%s/corner_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                plt.close('all')
                print('Made corner plot %d' % j)
    return            

def run(params, y_data_test, siz_x_data, y_normscale, load_dir):
    """ Function for producing samples from pre-trained CVAE network
    """

    # USEFUL SIZES
    xsh1 = siz_x_data
    if params['by_channel'] == True:
        ysh0 = np.shape(y_data_test)[0]
        ysh1 = np.shape(y_data_test)[1]
    else:
        ysh0 = np.shape(y_data_test)[1]
        ysh1 = np.shape(y_data_test)[2]
    z_dimension = params['z_dimension']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']
    n_modes = params['n_modes']
    n_hlayers_r1 = len(params['n_weights_r1'])
    n_hlayers_r2 = len(params['n_weights_r2'])
    n_hlayers_q = len(params['n_weights_q'])
    n_conv_r1 = len(params['n_filters_r1'])
    n_conv_r2 = len(params['n_filters_r2'])
    n_conv_q = len(params['n_filters_q'])
    n_filters_r1 = params['n_filters_r1']
    n_filters_r2 = params['n_filters_r2']
    n_filters_q = params['n_filters_q']
    filter_size_r1 = params['filter_size_r1']
    filter_size_r2 = params['filter_size_r2']
    filter_size_q = params['filter_size_q']
    batch_norm = params['batch_norm']
    ysh_conv_r1 = ysh1
    ysh_conv_r2 = ysh1
    ysh_conv_q = ysh1
    drate = params['drate']
    maxpool_r1 = params['maxpool_r1']
    maxpool_r2 = params['maxpool_r2']
    maxpool_q = params['maxpool_q']
    conv_strides_r1 = params['conv_strides_r1']
    conv_strides_r2 = params['conv_strides_r2']
    conv_strides_q = params['conv_strides_q']
    pool_strides_r1 = params['pool_strides_r1']
    pool_strides_r2 = params['pool_strides_r2']
    pool_strides_q = params['pool_strides_q']
    if n_filters_r1 != None:
        if params['by_channel'] == True:
            num_det = np.shape(y_data_test)[2]
        else:
            num_det = ysh0
    else:
        num_det = None

    # identify the indices of wrapped and non-wrapped parameters - clunky code
    wrap_mask, nowrap_mask, idx_mask = get_wrap_index(params)
    wrap_len = np.sum(wrap_mask)
    nowrap_len = np.sum(nowrap_mask)
   
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))

        # PLACEHOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder
        if n_filters_r1 != None:
            if params['by_channel'] == True:
                y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1, num_det], name="y_ph")
            else:
                y_ph = tf.placeholder(dtype=tf.float32, shape=[None, num_det, ysh1], name="y_ph")
        else:
            y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph")


        # LOAD VICI NEURAL NETWORKS
        r2_xzy = VICI_decoder.VariationalAutoencoder('VICI_decoder', wrap_mask, nowrap_mask, n_input1=z_dimension, 
                                                     n_input2=ysh_conv_r2, n_output=xsh1, n_weights=n_weights_r2, 
                                                     n_hlayers=n_hlayers_r2, drate=drate, n_filters=n_filters_r2, 
                                                     filter_size=filter_size_r2, maxpool=maxpool_r2, n_conv=n_conv_r2, 
                                                     conv_strides=conv_strides_r2, pool_strides=pool_strides_r2,num_det=num_det,batch_norm=batch_norm,by_channel=params['by_channel'], weight_init=params['weight_init'])
        r1_zy = VICI_encoder.VariationalAutoencoder('VICI_encoder', n_input=ysh_conv_r1, n_output=z_dimension, n_weights=n_weights_r1,   # generates params for r1(z|y)
                                                    n_modes=n_modes, n_hlayers=n_hlayers_r1, drate=drate, n_filters=n_filters_r1, 
                                                    filter_size=filter_size_r1, maxpool=maxpool_r1, n_conv=n_conv_r1, 
                                                    conv_strides=conv_strides_r1, pool_strides=pool_strides_r1, num_det=num_det,batch_norm=batch_norm,by_channel=params['by_channel'], weight_init=params['weight_init'])
        q_zxy = VICI_VAE_encoder.VariationalAutoencoder('VICI_VAE_encoder', n_input1=xsh1, n_input2=ysh_conv_q, n_output=z_dimension, 
                                                        n_weights=n_weights_q, n_hlayers=n_hlayers_q, drate=drate, 
                                                        n_filters=n_filters_q, filter_size=filter_size_q, maxpool=maxpool_q, n_conv=n_conv_q,conv_strides=conv_strides_q, pool_strides=pool_strides_q,num_det=num_det,batch_norm=batch_norm,by_channel=params['by_channel'], weight_init=params['weight_init'])  

        # reduce the y data size
        y_conv = y_ph

        # GET r1(z|y)
        r1_loc, r1_scale, r1_weight = r1_zy._calc_z_mean_and_sigma(y_conv)
        r1_scale = tf.sqrt(SMALL_CONSTANT + tf.exp(r1_scale))
        r1_weight = tf.squeeze(r1_weight)


        # define the r1(z|y) mixture model
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_weight),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_loc,
                          scale_diag=r1_scale))


        # DRAW FROM r1(z|y)
        r1_zy_samp = bimix_gauss.sample()

        # GET r2(x|z,y) from r(z|y) samples
        reconstruction_xzy = r2_xzy.calc_reconstruction(r1_zy_samp,y_conv)
        r2_xzy_mean_nowrap = reconstruction_xzy[0]
        r2_xzy_log_sig_sq_nowrap = reconstruction_xzy[1]
        if np.sum(wrap_mask)>0:
            r2_xzy_mean_wrap = reconstruction_xzy[2]
            r2_xzy_log_sig_sq_wrap = reconstruction_xzy[3]

        # draw from r2(x|z,y)
        r2_xzy_samp_gauss = q_zxy._sample_from_gaussian_dist(tf.shape(y_conv)[0], nowrap_len, r2_xzy_mean_nowrap, tf.log(SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_nowrap)))
        if np.sum(wrap_mask)>0:
            con = tf.reshape(tf.math.reciprocal(SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_wrap)),[-1,wrap_len])   # modelling wrapped scale output as log variance
            von_mises = tfp.distributions.VonMises(loc=2.0*np.pi*(r2_xzy_mean_wrap-0.5), concentration=con)
            r2_xzy_samp_vm = von_mises.sample()/(2.0*np.pi) + 0.5   # shift and scale from -pi-pi to 0-1
            r2_xzy_samp = tf.concat([tf.reshape(r2_xzy_samp_gauss,[-1,nowrap_len]),tf.reshape(r2_xzy_samp_vm,[-1,wrap_len])],1)
            r2_xzy_samp = tf.gather(r2_xzy_samp,tf.constant(idx_mask),axis=1)
            r2_xzy_loc = tf.concat([tf.reshape(r2_xzy_mean_nowrap,[-1,nowrap_len]),tf.reshape(r2_xzy_mean_wrap,[-1,wrap_len])],1)
            r2_xzy_loc = tf.gather(r2_xzy_loc,tf.constant(idx_mask),axis=1)
            r2_xzy_scale = tf.concat([tf.reshape(r2_xzy_log_sig_sq_nowrap,[-1,nowrap_len]),tf.reshape(r2_xzy_log_sig_sq_wrap,[-1,wrap_len])],1)
            r2_xzy_scale = tf.gather(r2_xzy_scale,tf.constant(idx_mask),axis=1)
        else:
            r2_xzy_loc = r2_xzy_mean_nowrap
            r2_xzy_scale = r2_xzy_log_sig_sq_nowrap
            r2_xzy_samp = r2_xzy_samp_gauss


        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VICI")]

        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(var_list_VICI)
        saver_VICI.restore(session,load_dir)

    # ESTIMATE TEST SET RECONSTRUCTION PER-PIXEL APPROXIMATE MARGINAL LIKELIHOOD and draw from q(x|y)
    ns = params['n_samples'] # number of samples to save per reconstruction

    if n_filters_r1 != None:
        y_data_test_exp = np.tile(y_data_test,(ns,1,1))/y_normscale
    else:
        y_data_test_exp = np.tile(y_data_test,(ns,1))/y_normscale
    run_startt = time.time()
    xs, loc, scale, mode_weights = session.run([r2_xzy_samp,r2_xzy_loc,r2_xzy_scale,r1_weight],feed_dict={y_ph:y_data_test_exp})
    run_endt = time.time()

    return xs, loc, scale, (run_endt - run_startt), mode_weights

