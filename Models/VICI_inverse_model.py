################################################################################################################
#
# --Variational Inference for Computational Imaging (VICI) Inverse Problem--
# 
# This model takes as input measured signals and infers target images/objects.
# There are two functions: 'train' and 'run'.
# 
# -train-
# takes as inputs a set of training target images/objects and a set of associated low fidelity signal predictions
# (e.g. from simulations). It then uses a forward model pre-trained with 'VICI_forward_model.train' and generates
# observations. With the generated observations, it learns an model to infer an approximation to the posterior of 
# images/objects given an observation. 
#
# INPUTS:
# params - list of parameters (see example for the necessary parameters).
# x_data - 2D array of training images/objects, where different samples are along dimension 0 and
#          values (e.g. pixel values) are along dimension 1.
# y_data_h - 2D array of training high fidelity observations, where different samples are along 
#            dimension 0 and values are along dimension 1. dimension 0 must be the same as that of
#             x_data (same number of samples as high fidelity observations).
# y_data_l - 2D array of training low fidelity observations, where different samples are along 
#            dimension 0 and values are along dimension 1. dimension 0 must be the same as that of
#             x_data (same number of samples as low fidelity observations).
# save_dir - directory in which to save the weights of the trained model.
#          
# OUTPUTS: 
# COST_PLOT - a vector containing the values of the cost function evaluated over a training subset
#             at intervals specified by the parameter 'report_interval'.
# KL_PLOT - a vector containing the values of the KL divergence evaluated over a training subset
#           at intervals specified by the parameter 'report_interval'.
#
# -run-
# loads the wights trained with 'train', takes as inputs a set of observed signals and returns samples 
# from the approximate posterior, an empirical mean and an empirical standard deviation.
# 
# INPUTS:
# params - list of parameters (see example for the necessary parameters).
# y_data_test - 2D array of observations, where different samples are along dimension
#               0 and values are along dimension 1.
# siz_x_data - dimensionality of target images/objects.
# load_dir - directory from which to load the weights of a trained model (trained with 'train').
#
# OUTPUTS:
# xm - 2D array of approximate posterior empirical means, where different samples are along dimension 0.
# xsx - 2D array of approximate posterior empirical stds, where different samples are along dimension 0.    
# XS - 3D array of draws from the approximate posterior, where different samples (reconstructions from
# different signals) are along dimension 0 and different draws from the same posterior are along dimension 2.
#
################################################################################################################

import numpy as np
import time
import tensorflow as tf
import tensorflow_probability as tfp
import corner

#from Neural_Networks import OELBO_decoder_difference
#from Neural_Networks import OELBO_encoder
#from Neural_Networks import VAE_encoder
from Neural_Networks import VICI_decoder
from Neural_Networks import VICI_encoder
from Neural_Networks import VICI_VAE_encoder
from Neural_Networks import batch_manager
from Models import VICI_inverse_model
from data import chris_data
import plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from data import make_samples

tfd = tfp.distributions

# NORMALISE DATASET FUNCTION
def tf_normalise_dataset(xp):
    
    Xs = tf.shape(xp)
    
    l2norm = tf.sqrt(tf.reduce_sum(tf.multiply(xp,xp),1))
    l2normr = tf.reshape(l2norm,[Xs[0],1])
    x_data = tf.divide(xp,l2normr)
  
    # comment this if you want to use normalise 
    x_data = xp 
    return x_data

# MULTIMODAL UPDATE: NORMALISE DATASET TO THE SUM FUNCTION
def tf_normalise_sum_dataset(xp):
    
    Xs = tf.shape(xp)
    
    log_norm = tf.math.reduce_logsumexp(xp,1)
    #norm = tf.reshape(tf.reduce_sum(xp,1),[Xs[0],1])
    log_norm = tf.reshape(log_norm,[Xs[0],1])
    #x_data = tf.divide(xp,norm)
    x_data = tf.add(xp,-log_norm)    

    return x_data

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

def train(params, x_data, y_data, x_data_test, y_data_test, y_data_test_noisefree, y_normscale, save_dir, truth_test, bounds, fixed_vals, posterior_truth_test):    

    # if True, do multi-modal
    multi_modal = True

    # USEFUL SIZES
    xsh = np.shape(x_data)
    ysh = np.shape(y_data)[1]
    z_dimension = params['z_dimension']
    bs = params['batch_size']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']
    ramp_start = 1e3
    ramp_end = 1e4

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
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh], name="y_ph")    # data placeholder
        idx = tf.placeholder(tf.int32)

        # LOAD VICI NEURAL NETWORKS
        r2_xzy = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh[1], z_dimension+ysh, n_weights_r2, wrap_mask, nowrap_mask) # r2(x|z,y)
        r1_zy_a = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh, z_dimension, n_weights_r1) # generates params for r1(z|y)
        r1_zy_b = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh, z_dimension, n_weights_r1) # generates params for r1(z|y)
        r1_zy_c = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh, z_dimension, n_weights_r1) # generates params for r1(z|y)
        r1_zy_d = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh, z_dimension, n_weights_r1) # generates params for r1(z|y)
        q_zxy = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh[1]+ysh, z_dimension, n_weights_q) # used to sample from q(z|x,y)?
        tf.set_random_seed(np.random.randint(0,10))

          
        SMALL_CONSTANT = 1e-6
        ramp_start = 2e4
        ramp_stop = 1e6
        #ramp = tf.math.minimum(1.0,(tf.dtypes.cast(idx,dtype=tf.float32)/1.0e5)**(3.0))         
        #ramp = 1.0 - 1.0/tf.sqrt(1.0 + (tf.dtypes.cast(idx,dtype=tf.float32)/1000.0))
        ramp = (tf.log(tf.dtypes.cast(idx,dtype=tf.float32)) - tf.log(ramp_start))/(tf.log(ramp_stop)-tf.log(ramp_start))
        ramp = tf.minimum(tf.math.maximum(0.0,ramp),1.0)
        #ramp=1.0
        if multi_modal == False:
            ramp = 1.0

        # GET r1(z|y)
        # run inverse autoencoder to generate mean and logvar of z given y data - these are the parameters for r1(z|y)
        r1_zy_mean_a, r1_zy_log_sig_sq_a, r1_zy_wa = r1_zy_a._calc_z_mean_and_sigma(y_ph)        
        r1_zy_mean_b, r1_zy_log_sig_sq_b, r1_zy_wb = r1_zy_b._calc_z_mean_and_sigma(y_ph)
        r1_zy_mean_c, r1_zy_log_sig_sq_c, r1_zy_wc = r1_zy_c._calc_z_mean_and_sigma(y_ph)
        r1_zy_mean_d, r1_zy_log_sig_sq_d, r1_zy_wd = r1_zy_d._calc_z_mean_and_sigma(y_ph)
        if multi_modal == True:
            r1_zy_locs = tf.stack([r1_zy_mean_a,r1_zy_mean_b,r1_zy_mean_c,r1_zy_mean_d],axis=1)
            r1_zy_scales = tf.stack([tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_a)),tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_b)), tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_c)),tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_d))],axis=1)
        #r1_zy_log_weights = tf_normalise_sum_dataset(r1_zy_log_weights)
            r1_zy_log_weights = tf.concat([r1_zy_wa,r1_zy_wb,r1_zy_wc,r1_zy_wd],1)
        else:
            r1_zy_locs = tf.stack([r1_zy_mean_a],axis=1)
            r1_zy_scales = tf.stack([tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_a))],axis=1)
            r1_zy_log_weights = tf.concat([r1_zy_wa],1)
        r1_zy_log_weights = tf_normalise_sum_dataset(r1_zy_log_weights)

        
        # define the r1(z|y) mixture model
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_zy_log_weights),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_zy_locs,
                          scale_diag=r1_zy_scales))


        # DRAW FROM r1(z|y) - given the Gaussian parameters generate z samples
        r1_zy_samp = bimix_gauss.sample()        
        
        # GET q(z|x,y)
        q_zxy_mean, q_zxy_log_sig_sq = q_zxy._calc_z_mean_and_sigma(tf.concat([x_ph,y_ph],axis=1))
        #q_zxy_mean, q_zxy_log_sig_sq = q_zxy._calc_z_mean_and_sigma(x_ph)

        # DRAW FROM q(z|x,y)
        q_zxy_samp = q_zxy._sample_from_gaussian_dist(bs_ph, z_dimension, q_zxy_mean, tf.log(SMALL_CONSTANT + tf.exp(q_zxy_log_sig_sq)))
        
        # GET r2(x|z,y)
        reconstruction_xzy = r2_xzy.calc_reconstruction(tf.concat([q_zxy_samp,y_ph],axis=1))
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
            #kappa = tf.math.reciprocal(SMALL_CONSTANT + r2_xzy_log_sig_sq_wrap)
            #reconstr_loss_vm_num = tf.multiply(kappa,tf.math.cos(2.0*np.pi*(r2_xzy_mean_wrap - tf.boolean_mask(x_ph,wrap_mask,axis=1))))
            #reconstr_loss_vm_denum = -np.log(2.0*np.pi) - tf.log(tf.math.bessel_i0(kappa))
            #reconstr_loss_vm = tf.reduce_sum(reconstr_loss_vm_num + reconstr_loss_vm_denum,axis=1)
            con = tf.reshape(tf.math.reciprocal(SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_wrap)),[-1,wrap_len])   # modelling wrapped scale output as log variance
            von_mises = tfp.distributions.VonMises(loc=2.0*np.pi*tf.reshape(r2_xzy_mean_wrap,[-1,wrap_len]), concentration=con)   # define p_vm(2*pi*mu,con=1/sig^2)
            reconstr_loss_vm = tf.reduce_sum(von_mises.log_prob(2.0*np.pi*tf.reshape(tf.boolean_mask(x_ph,wrap_mask,axis=1),[-1,wrap_len])),axis=1)   # 2pi is the von mises input range
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
    for i in range(params['num_iterations']):

        next_indices = indices_generator.next_indices()

        # Make noise realizations and add to training data
        next_x_data = x_data[next_indices,:]
        next_y_data = y_data[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],int(params['ndata']*len(fixed_vals['det']))))
        next_y_data /= y_normscale  # required for fast convergence

        # train to minimise the cost function
        session.run(minimize, feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, idx:i})

        # if we are in a report iteration extract cost function values
        if i % params['report_interval'] == 0 and i > 0:

            cost, kl, AB_batch = session.run([cost_R, KL, r1_zy_log_weights], feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, idx:i})
            plotdata.append([cost,kl,cost+kl])

            if params['print_values']==True:
                print('--------------------------------------------------------------')
                print('Iteration:',i)
                print('Training Set -ELBO:',cost)
                print('Approx KL Divergence:',kl)
                print('Total cost:',kl + cost) 

        if i % params['save_interval'] == 0 and i > 0:

            # Save model 
            save_path = saver.save(session,save_dir)

        if i % params['plot_interval'] == 0 and i>0:
            
            # use the testing data for some plots
            for j in range(params['r']*params['r']):

                # make spcific data for plots that contains a training data sample with lots of different noise
                x_data_zplot = np.tile(x_data_test[j,:],(params['n_samples'],1))
                y_data_zplot = np.tile(y_data_test[j,:],(params['n_samples'],1))
                y_data_zplot += np.random.normal(0,1,size=(params['n_samples'],params['ndata']))
                y_data_zplot /= y_normscale  # required for fast convergence                
                
                # run a training pass and extract parameters (do it multiple times for ease of reading)
                # get q(z) data
                q_z_plot_data, q_z_log_sig_sq_data = session.run([q_zxy_mean,q_zxy_log_sig_sq], feed_dict={bs_ph:params['n_samples'], x_ph:x_data_zplot, y_ph:y_data_zplot, idx:i})
          
                # get r1(z) data
                r1_z_locs, r1_z_scales, r1_samp, r1_z_weights_plot_data = session.run([r1_zy_locs,r1_zy_scales,r1_zy_samp,r1_zy_log_weights], feed_dict={bs_ph:params['n_samples'], x_ph:x_data_zplot, y_ph:y_data_zplot, idx:i})
                
                # get r2(x) data
                r2_loc, r2_scale = session.run([r2_xzy_mean, r2_xzy_scale], feed_dict={bs_ph:params['n_samples'], x_ph:x_data_zplot, y_ph:y_data_zplot, idx:i})
                #print('<r2 mean nowrap> = {}'.format(np.mean(r2_mean_nowrap)))
                #print('<r2 mean wrap> = {}'.format(np.mean(r2_mean_wrap)))
                #print('<r2 sigsq nowrap> = {}'.format(np.mean(r2_log_sig_sq_nowrap)))
                #print('<r2 sigsq wrap> = {}'.format(np.mean(r2_log_sig_sq_wrap)))



                """ 
                try:
                    # Make corner plot of latent space samples from the q distribution
                    figure = corner.corner(q_z_plot_data, #labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84])
                       #range=[[-2,2]]*np.shape(x_data_test)[1])
                       #truths=x_data_test[j,:],
                       #show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/qz_mean_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/qz_mean_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))           
                    plt.close()
                except:
                    pass

                try:
                    # Make corner plot of latent space samples from the q distribution
                    figure = corner.corner(q_z_log_sig_sq_data, #labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84])
                       #range=[[-2,2]]*np.shape(x_data_test)[1])
                       #truths=x_data_test[j,:],
                       #show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/qz_log_sig_sq_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/qz_log_sig_sq_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                    plt.close()
                except:
                    pass

                try:
                    # Make corner plot of latent space samples from the 1 distribution
                    figure = corner.corner(np.concatenate([r1_z_locs[:,0],r1_z_locs[:,1],r1_z_locs[:,2]],0), #labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84])
                       #range=[[-2,2]]*np.shape(x_data_test)[1])
                       #truths=x_data_test[j,:],
                       #show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/r1z_mean_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/r1z_mean_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                    plt.close()
                except:
                    pass

                try:
                    # Make corner plot of latent space samples from the q distribution
                    figure = corner.corner(np.concatenate([r1_z_scales[:,0],r1_z_scales[:,1],r1_z_scales[:,2]],0), #labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84])
                       #range=[[0,1]]*np.shape(x_data_test)[1],
                       #truths=x_data_test[j,:],
                       #show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/r1z_log_sig_sq_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/r1z_log_sig_sq_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                    plt.close()
                except:
                    pass

                try:
                    # Make corner plot of latent space samples from the q distribution
                    figure = corner.corner(q_samp, #labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84])
                       #range=[[-2,2]]*np.shape(x_data_test)[1])
                       #truths=x_data_test[j,:],
                       #show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/qz_samp_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/qz_samp_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                    plt.close()
                except:
                    pass

                try:
                    # Make corner plot of latent space samples from the q distribution
                    figure = corner.corner(r1_samp, #labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84])
                       #range=[[-2,2]]*np.shape(x_data_test)[1])
                       #truths=x_data_test[j,:],
                       #show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/r1z_samp_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/r1z_samp_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                    plt.close()
                except:
                    pass

                try:
                    # Make corner plot of latent space samples from the q distribution
                    figure = corner.corner(r2_loc, labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84],
                       #range=[[-0.1,1.1]]*np.shape(x_data_test)[1],
                       truths=x_data_zplot[j,:])
                       #show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/r2x_mean_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/r2x_mean_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j)) 
                    plt.close()
                except:
                    pass
 
                try:
                    # Make corner plot of latent space samples from the q distribution
                    figure = corner.corner(r2_scale, labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84])
                       #range=[[-2,2]]*np.shape(x_data_test)[1])
                       #truths=x_data_test[j,:],
                       #show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/r2x_log_sig_sq_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/r2x_log_sig_sq_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                    plt.close()
                except:
                    pass
 
                #try:
                #    # Make corner plot of latent space samples from the q distribution
                #    figure = corner.corner(r2_mean_testpath, #labels=params['inf_pars'],
                #       quantiles=[0.16, 0.5, 0.84],
                #       range=[[-0.1,1.1]]*np.shape(x_data_test)[1],
                #       truths=x_data_zplot[j,:])
                #       #show_titles=True, title_kwargs={"fontsize": 12})
                #    plt.savefig('%s/r2x_mean_testpath_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                #    plt.savefig('%s/latest_%s/r2x_mean_testpath_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                #    plt.close()
                #except:
                #    pass

                #try:
                #    # Make corner plot of latent space samples from the q distribution
                #    figure = corner.corner(r2_log_sig_sq_testpath, #labels=params['inf_pars'],
                #       quantiles=[0.16, 0.5, 0.84])
                #       #range=[[-2,2]]*np.shape(x_data_test)[1])
                #       #truths=x_data_test[j,:],
                #       #show_titles=True, title_kwargs={"fontsize": 12})
                #    plt.savefig('%s/r2x_log_sig_sq_testpath_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                #    plt.savefig('%s/latest_%s/r2x_log_sig_sq_testpath_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                #    plt.close()
                #except:
                #    pass

                #try:
                #    # Make corner plot of latent space samples from the q distribution
                #    figure = corner.corner(r2_samp_testpath, #labels=params['inf_pars'],
                #       quantiles=[0.16, 0.5, 0.84],
                #       range=[[-0.1,1.1]]*np.shape(x_data_test)[1],
                #       truths=x_data_zplot[j,:])
                #       #show_titles=True, title_kwargs={"fontsize": 12})
                #    plt.savefig('%s/r2x_samp_testpath_%s_train%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                #    plt.savefig('%s/latest_%s/r2x_samp_testpath_%s_train%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                #    plt.close()
                #except:
                #    pass

                # plot the AB histogram
                density_flag = False
                plt.figure()
                plt.hist(r1_z_weights_plot_data[:,0],25,alpha=0.5,density=density_flag,label='component 0')
                plt.hist(r1_z_weights_plot_data[:,1],25,alpha=0.5,density=density_flag,label='component 1')
                plt.hist(r1_z_weights_plot_data[:,2],25,alpha=0.5,density=density_flag,label='component 2')
                plt.xlabel('iteration')
                plt.ylabel('KL')
                plt.legend()
                plt.savefig('%s/mixweights_%s_train%d_%d_linear.png' % (params['plot_dir'],params['run_label'],j,i))
                plt.savefig('%s/latest_%s/mixweights_%s_train%d_linear.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                plt.close()
                """
                

            # just run the network on the test data
            for j in range(params['r']*params['r']):

                # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                XS, loc, scale, dt  = VICI_inverse_model.run(params, y_data_test[j].reshape([1,-1]), np.shape(x_data_test)[1],
                                                 y_normscale, 
                                                 "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
                print('Runtime to generate {} samples = {} sec'.format(params['n_samples'],dt))            
               
                # Generate final results plots
                plotter = plots.make_plots(params,posterior_truth_test,np.expand_dims(XS, axis=0),np.expand_dims(truth_test[j],axis=0))

                # Make corner plots
                plotter.make_corner_plot(y_data_test_noisefree[j,:params['ndata']],y_data_test[j,:params['ndata']],bounds,j,i,sampler='dynesty1')

                del plotter

                """
                try:
                    
                    # Make Chris corner plot of VItamin posterior samples
                    figure = corner.corner(XS, labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84],
                       range=[[0.0,1.0]]*np.shape(x_data_test)[1],
                       truths=x_data_test[j,:],
                       show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/loc_output_%s_test%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/loc_output_%s_test%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                    plt.close()
                except:
                    pass
 
                try:
                    # Make corner plot of VItamin posterior scale params
                    figure = corner.corner(scale, labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84],
                       #range=[[-0.1,1.1]]*np.shape(x_data_test)[1],
                       #truths=x_data_test[j,:],
                       show_titles=True, title_kwargs={"fontsize": 12})
                    plt.savefig('%s/scale_output_%s_test%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                    plt.savefig('%s/latest_%s/scale_output_%s_test%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                    plt.close()
                    
                except Exception as e:
                    print(e)
                    exit()
                    exit()
                    exi(t)
                    pass 
                """


            # Make loss plot
            try:
                plt.figure()
                xvec = params['report_interval']*np.arange(np.array(plotdata).shape[0])
                plt.semilogx(xvec,np.array(plotdata)[:,0],label='recon')
                plt.semilogx(xvec,np.array(plotdata)[:,1],label='KL')
                plt.semilogx(xvec,np.array(plotdata)[:,2],label='total')
                #plt.ylim([-15,12])
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.legend()
                plt.savefig('%s/cost_%s.png' % (params['plot_dir'],params['run_label']))
                plt.ylim([np.min(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,0]), np.max(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,1])])
                plt.savefig('%s/cost_zoom_%s.png' % (params['plot_dir'],params['run_label']))
                plt.close()
            except:
                 pass            

            
            # plot the AB histogram
            try:
                density_flag = False
                plt.figure()
                plt.hist(np.exp(AB_batch[:,0]),25,density=density_flag,label='component 0')
                plt.hist(np.exp(AB_batch[:,1]),25,density=density_flag,label='component 1')
                plt.hist(np.exp(AB_batch[:,2]),25,density=density_flag,label='component 2')
                plt.hist(np.exp(AB_batch[:,3]),25,density=density_flag,label='component 3')

                plt.xlabel('iteration')
                plt.ylabel('KL')
                plt.legend()
                plt.savefig('%s/mixweights_%s_batch_%d_linear.png' % (params['plot_dir'],params['run_label'],i))
                plt.savefig('%s/latest_%s/mixweights_%s_batch_linear.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.close()
            except:
                pass

            # plot the AB histogram

            try:
                plt.figure()
                plt.hist(AB_batch[:,0],25,density=density_flag,label='component 0')
                plt.hist(AB_batch[:,1],25,density=density_flag,label='component 1')
                plt.hist(AB_batch[:,0],25,density=density_flag,label='component 2')
                plt.hist(AB_batch[:,1],25,density=density_flag,label='component 3')
                plt.xlabel('Mixture weight')
                plt.ylabel('p(w)')
                plt.legend()
                plt.savefig('%s/mixweights_%s_batch_%d_log.png' % (params['plot_dir'],params['run_label'],i))
                plt.savefig('%s/latest_%s/mixweights_%s_batch_log.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.close()
            except:
                pass
            

    return            

def run(params, y_data_test, siz_x_data, y_normscale, load_dir):

    multi_modal = True

    # USEFUL SIZES
    xsh1 = siz_x_data
    ysh0 = np.shape(y_data_test)[0]
    ysh1 = np.shape(y_data_test)[1]

    z_dimension = params['z_dimension']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']

    # identify the indices of wrapped and non-wrapped parameters - clunky code
    wrap_mask, nowrap_mask, idx_mask = get_wrap_index(params)
    wrap_len = np.sum(wrap_mask)
    nowrap_len = np.sum(nowrap_mask)
   
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-8

        # PLACEHOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph")


        # LOAD VICI NEURAL NETWORKS
        r2_xzy = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh1, z_dimension+ysh1, n_weights_r2,wrap_mask,nowrap_mask)
        r1_zy_a = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights_r1)
        r1_zy_b = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights_r1)
        r1_zy_c = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights_r1)
        r1_zy_d = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights_r1)
        q_zxy = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh1+ysh1, z_dimension, n_weights_q)

        # GET r1(z|y)
        r1_zy_mean_a, r1_zy_log_sig_sq_a, r1_zy_wa = r1_zy_a._calc_z_mean_and_sigma(y_ph)
        r1_zy_mean_b, r1_zy_log_sig_sq_b, r1_zy_wb = r1_zy_b._calc_z_mean_and_sigma(y_ph)
        r1_zy_mean_c, r1_zy_log_sig_sq_c, r1_zy_wc = r1_zy_c._calc_z_mean_and_sigma(y_ph)
        r1_zy_mean_d, r1_zy_log_sig_sq_d, r1_zy_wd = r1_zy_d._calc_z_mean_and_sigma(y_ph)
        r1_zy_weights = 0.0*tf.concat([r1_zy_wa, r1_zy_wb, r1_zy_wc, r1_zy_wd],1)
        r1_zy_locs = tf.stack([r1_zy_mean_a,r1_zy_mean_b,r1_zy_mean_c,r1_zy_mean_d],axis=1)
        r1_zy_scales = tf.stack([tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_a)),tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_b)),tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_c)),tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_d))],axis=1)
        #r1_zy_weights = tf_normalise_sum_dataset(r1_zy_weights)

        means = tf.stack([r1_zy_mean_a,r1_zy_mean_b,r1_zy_mean_c,r1_zy_mean_d],axis=1)
        scales = tf.stack([tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_a)),tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_b)),tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_c)),tf.sqrt(SMALL_CONSTANT + tf.exp(r1_zy_log_sig_sq_d))],axis=1)

        # define the r1(z|y) mixture model
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_zy_weights),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_zy_locs,
                          scale_diag=r1_zy_scales))


        # DRAW FROM r1(z|y)
        r1_zy_samp = bimix_gauss.sample()

        # GET r2(x|z,y) from r(z|y) samples
        reconstruction_xzy = r2_xzy.calc_reconstruction(tf.concat([r1_zy_samp,y_ph],1))
        r2_xzy_mean_nowrap = reconstruction_xzy[0]
        r2_xzy_log_sig_sq_nowrap = reconstruction_xzy[1]
        if np.sum(wrap_mask)>0:
            r2_xzy_mean_wrap = reconstruction_xzy[2]
            r2_xzy_log_sig_sq_wrap = reconstruction_xzy[3]

        # draw from r2(x|z,y)
        r2_xzy_samp_gauss = q_zxy._sample_from_gaussian_dist(tf.shape(y_ph)[0], nowrap_len, r2_xzy_mean_nowrap, tf.log(SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_nowrap)))
        if np.sum(wrap_mask)>0:
            var = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_wrap)     # modelling wrapped scale output as a variance
            von_mises = tfp.distributions.VonMises(loc=2.0*np.pi*r2_xzy_mean_wrap, concentration=tf.math.reciprocal(var))
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

    y_data_test_exp = np.tile(y_data_test,(ns,1))/y_normscale
    run_startt = time.time()
    #XS = session.run(r2_xzy_samp,feed_dict={y_ph:y_data_test_exp})
    xs, loc, scale = session.run([r2_xzy_samp,r2_xzy_loc,r2_xzy_scale],feed_dict={y_ph:y_data_test_exp})
    run_endt = time.time()

    return xs, loc, scale, (run_endt - run_startt)

def resume_training(params, x_data, y_data_l, siz_high_res, save_dir, train_files,normscales,y_data_train_noisefree,y_normscale):    

    x_data = x_data
    y_data_train_l = y_data_l
    
    # USEFUL SIZES
    xsh = np.shape(x_data)
    yshl1 = np.shape(y_data_l)[1]
    ysh1 = siz_high_res
    
    #z_dimension_fm = params['z_dimensions_fw']
    #n_weights_fm = params['n_weights_fw']
    
    z_dimension = params['z_dimension']
    bs = params['batch_size']
    n_weights = params['n_weights']
    lam = 1
    
    # Allow GPU growth 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph = tf.Graph()
    session = tf.Session(graph=graph,config=config)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-6
        
        # PLACE HOLDERS
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_ph")
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph") # batch size placeholder
        yt_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="yt_ph")
        
        # LOAD FORWARD MODEL NEURAL NETWORKS
        #DEC_XYlZtoYh = OELBO_decoder_difference.VariationalAutoencoder("OELBO_decoder", ysh1, z_dimension_fm+yshl1+xsh[1], n_weights_fm) # p(Yh|X,Yl,Z)
        #ENC_XYltoZ = OELBO_encoder.VariationalAutoencoder("OELBO_encoder", yshl1+xsh[1], z_dimension_fm, n_weights_fm) # p(Z|X,Yl)
        #ENC_XYhYltoZ = VAE_encoder.VariationalAutoencoder("vae_encoder", xsh[1]+ysh1+yshl1, z_dimension_fm, n_weights_fm) # q(Z|X,Yl,Yh)
        
        # LOAD VICI NEURAL NETWORKS
        autoencoder = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh[1], z_dimension+ysh1, n_weights)
        autoencoder_ENC = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights)
        autoencoder_VAE = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh[1]+ysh1, z_dimension, n_weights)
        
        # DEFINE MULTI-FIDELITY FORWARD MODEL
        #####################################################################################################################
        SMALL_CONSTANT = 1e-6
        
        # NORMALISE INPUTS
        yl_ph_n = tf_normalise_dataset(yt_ph)
        x_ph_n = tf_normalise_dataset(x_ph)
        #yl_ph_n = yt_ph
        #x_ph_n = x_ph
        
        # GET p(Z|X,Yl)
        #zxyl_mean,zxyl_log_sig_sq = ENC_XYltoZ._calc_z_mean_and_sigma(tf.concat([x_ph_n,yl_ph_n],1))
        #rxyl_samp = ENC_XYhYltoZ._sample_from_gaussian_dist(tf.shape(x_ph_n)[0], z_dimension_fm, zxyl_mean, tf.log(tf.exp(zxyl_log_sig_sq)+SMALL_CONSTANT))
        
        # GET p(Yh|X,Yl,Z) FROM SAMPLES Z ~ p(Z|X,Yl)
        #reconstruction_yh = DEC_XYlZtoYh.calc_reconstruction(tf.concat([x_ph_n,yl_ph_n,rxyl_samp],1))
        #yh_diff = reconstruction_yh[0]
        #yh_mean = yl_ph_n+yh_diff
        #yh_log_sig_sq = reconstruction_yh[1]
        #y = ENC_XYhYltoZ._sample_from_gaussian_dist(tf.shape(yt_ph)[0], tf.shape(yt_ph)[1], yh_mean, yh_log_sig_sq)
        
        # DRAW SYNTHETIC Ys TRAINING DATA
#        _, y = OM.forward_model(x_ph_amp,x_ph_ph)
        
        ##########################################################################################################################################
        
        # GET r(z|y)
        #y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph")
        #y_ph_n = tf_normalise_dataset(y_ph)
        #y_ph_n = y_ph
        zy_mean,zy_log_sig_sq = autoencoder_ENC._calc_z_mean_and_sigma(yl_ph_n)
        
        # DRAW FROM r(z|y)
        rzy_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension, zy_mean, zy_log_sig_sq)
        
        # GET r(x|z,y) from r(z|y) samples
        rzy_samp_y = tf.concat([rzy_samp,yl_ph_n],1)
        reconstruction_xzy = autoencoder.calc_reconstruction(rzy_samp_y)
        x_mean = reconstruction_xzy[0]
        x_log_sig_sq = reconstruction_xzy[1]
        
        # KL(r(z|y)||p(z))
        latent_loss = -0.5 * tf.reduce_sum(1 + zy_log_sig_sq - tf.square(zy_mean) - tf.exp(zy_log_sig_sq), 1)
        KL = tf.reduce_mean(latent_loss)
        
        # GET q(z|x,y)
        xy_ph = tf.concat([x_ph_n,yl_ph_n],1)
        zx_mean,zx_log_sig_sq = autoencoder_VAE._calc_z_mean_and_sigma(xy_ph)
        
        # DRAW FROM q(z|x,y)
        qzx_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension, zx_mean, zx_log_sig_sq)
        
        # GET r(x|z,y)
        qzx_samp_y = tf.concat([qzx_samp,yl_ph_n],1)
        reconstruction_xzx = autoencoder.calc_reconstruction(qzx_samp_y)
        x_mean_vae = reconstruction_xzx[0]
        x_log_sig_sq_vae = reconstruction_xzx[1]
        
        # COST FROM RECONSTRUCTION
        normalising_factor_x_vae = - 0.5 * tf.log(SMALL_CONSTANT+tf.exp(x_log_sig_sq_vae)) - 0.5 * np.log(2 * np.pi)
        square_diff_between_mu_and_x_vae = tf.square(x_mean_vae - x_ph_n)
        inside_exp_x_vae = -0.5 * tf.div(square_diff_between_mu_and_x_vae,SMALL_CONSTANT+tf.exp(x_log_sig_sq_vae))
        reconstr_loss_x_vae = -tf.reduce_sum(normalising_factor_x_vae + inside_exp_x_vae, 1) # Take sum along axis 1
        cost_R_vae = tf.reduce_mean(reconstr_loss_x_vae) # Take mean along the diagonal
        
        # KL(q(z|x,y)||r(z|y))
        v_mean = zy_mean #2
        aux_mean = zx_mean #1
        v_log_sig_sq = tf.log(tf.exp(zy_log_sig_sq)+SMALL_CONSTANT) #2
        aux_log_sig_sq = tf.log(tf.exp(zx_log_sig_sq)+SMALL_CONSTANT) #1
        v_log_sig = tf.log(tf.sqrt(tf.exp(v_log_sig_sq))) #2
        aux_log_sig = tf.log(tf.sqrt(tf.exp(aux_log_sig_sq))) #1
        cost_VAE_a = v_log_sig-aux_log_sig+tf.divide(tf.exp(aux_log_sig_sq)+tf.square(aux_mean-v_mean),2*tf.exp(v_log_sig_sq))-0.5
        cost_VAE_b = tf.reduce_sum(cost_VAE_a,1)
        KL_vae = tf.reduce_mean(cost_VAE_b)
        
        # THE VICI COST FUNCTION
        lam_ph = tf.placeholder(dtype=tf.float32, name="lam_ph")
        COST_VAE = KL_vae+cost_R_vae
        COST = COST_VAE
        
        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VICI")]
        var_list_ELBO = [var for var in tf.trainable_variables() if var.name.startswith("ELBO")]
        
        # DEFINE OPTIMISER (using ADAM here)
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate']) 
        minimize = optimizer.minimize(COST,var_list = var_list_VICI)
        
        # DRAW FROM q(x|y)
        qx_samp = autoencoder_ENC._sample_from_gaussian_dist(bs_ph, xsh[1], x_mean, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq)))
        
        # INITIALISE AND RUN SESSION
#        init = tf.variables_initializer(var_list_VICI)
        init = tf.initialize_all_variables()
        session.run(init)
        #saver_ELBO = tf.train.Saver(var_list_ELBO)
        #saver_ELBO.restore(session,load_dir)
        saver = tf.train.Saver(var_list_VICI)
        saver.restore(session,save_dir)
    
    KL_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1)) # vector to store test OELBO values
    COST_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1)) # vector to store test VAE ELBO values
    
    print('Training Inference Model...')    
    # START OPTIMISATION OF OELBO
    indices_generator = batch_manager.SequentialIndexer(params['batch_size'], xsh[0])
    ni = -1
    test_n = 100
    for i in range(params['num_iterations']):
        
        next_indices = indices_generator.next_indices()

        # Make 25 noise realizations
        if params['do_extra_noise']:
            x_data_train_l = x_data[next_indices,:]
            y_data_train_l = y_data_train_noisefree[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],params['ndata']))
            y_data_train_l /= y_normscale[0]
            #print('generated {} elements of new training data noise'.format(params['batch_size']))

            session.run(minimize, feed_dict={bs_ph:bs, x_ph:x_data_train_l, lam_ph:lam, yt_ph:y_data_train_l}) # minimising cost function
        else:       
            #yn = session.run(y, feed_dict={bs_ph:bs, x_ph:x_data[next_indices, :], yt_ph:y_data_train_l[next_indices, :]})
            #session.run(minimize, feed_dict={bs_ph:bs, x_ph:x_data[next_indices, :],  y_ph:yn, lam_ph:lam, yt_ph:y_data_train_l[next_indices, :]}) # minimising cost function
            session.run(minimize, feed_dict={bs_ph:bs, x_ph:x_data[next_indices, :], lam_ph:lam, yt_ph:y_data_train_l[next_indices, :]}) # minimising cost function
        
        if i % params['report_interval'] == 0:
                ni = ni+1
                
                #ynt = session.run(y, feed_dict={bs_ph:test_n, x_ph:x_data[0:test_n,:], yt_ph:y_data_train_l[0:test_n,:]})
                #cost_value_vae, KL_VAE = session.run([COST_VAE, KL_vae], feed_dict={bs_ph:test_n, x_ph:x_data[0:test_n,:], y_ph:ynt, lam_ph:lam, yt_ph:y_data_train_l[0:test_n,:]})
                cost_value_vae, KL_VAE = session.run([COST_VAE, KL_vae], feed_dict={bs_ph:test_n, x_ph:x_data[0:test_n,:], lam_ph:lam, yt_ph:y_data_train_l[0:test_n,:]})
                KL_PLOT[ni] = KL_VAE
                COST_PLOT[ni] = -cost_value_vae

                # plot losses
                plotter.make_loss_plot(COST_PLOT[:ni+1],KL_PLOT[:ni+1],params['report_interval'],fwd=False)
                
                if params['print_values']==True:
                    print('--------------------------------------------------------------')
                    print('Iteration:',i)
                    print('Training Set ELBO:',-cost_value_vae)
                    print('KL Divergence:',KL_VAE)
       
        if i % params['plot_interval'] == 0 and i>0:
            # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
            _, _, XS, _  = VICI_inverse_model.run(params, y_data_test_h, np.shape(x_data_train)[1], "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])

            # Convert XS back to unnormalized version
            if params['do_normscale']:
                for m in range(params['ndim_x']):
                    XS[:,m,:] = XS[:,m,:]*normscales[m]

            # Make KL plot
            plotter.gen_kl_plots(VICI_inverse_model,y_data_test_h,x_data_train,normscales)

            # Make corner plots
            plotter.make_corner_plot(sampler='dynesty1')

            # Make KL plot
            plotter.gen_kl_plots(VICI_inverse_model,y_data_test_h,x_data_train,normscales)

            # Make pp plot
#            plotter.plot_pp(VICI_inverse_model,y_data_train_l,x_data_train,0,normscales)
            

        if i % params['save_interval'] == 0:
             
            save_path = saver.save(session,save_dir)
                
                
    return COST_PLOT, KL_PLOT, train_files

def compute_ELBO(params, x_data, y_data_h, load_dir):
    
    # USEFUL SIZES
    xsh = np.shape(x_data)
    ysh1 = np.shape(y_data_h)[1]
    
    z_dimension = params['z_dimension']
    n_weights = params['n_weights']
    lam = 1
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-6
        
        # PLACE HOLDERS
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_ph")
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph") # batch size placeholder
        
        # LOAD VICI NEURAL NETWORKS
        autoencoder = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh[1], z_dimension+ysh1, n_weights)
        autoencoder_ENC = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights)
        autoencoder_VAE = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh[1]+ysh1, z_dimension, n_weights)
        
        # GET r(z|y)
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph")
        y_ph_n = tf_normalise_dataset(y_ph)
        x_ph_n = tf_normalise_dataset(x_ph)
        #y_ph_n = y_ph
        #x_ph_n = x_ph
        zy_mean,zy_log_sig_sq = autoencoder_ENC._calc_z_mean_and_sigma(y_ph_n)
        
        # DRAW FROM r(z|y)
        rzy_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension, zy_mean, zy_log_sig_sq)
        
        # GET r(x|z,y) from r(z|y) samples
        rzy_samp_y = tf.concat([rzy_samp,y_ph_n],1)
        reconstruction_xzy = autoencoder.calc_reconstruction(rzy_samp_y)
        x_mean = reconstruction_xzy[0]
        x_log_sig_sq = reconstruction_xzy[1]
        
        # KL(r(z|y)||p(z))
        latent_loss = -0.5 * tf.reduce_sum(1 + zy_log_sig_sq - tf.square(zy_mean) - tf.exp(zy_log_sig_sq), 1)
        KL = tf.reduce_mean(latent_loss)
        
        # GET q(z|x,y)
        xy_ph = tf.concat([x_ph_n,y_ph_n],1)
        zx_mean,zx_log_sig_sq = autoencoder_VAE._calc_z_mean_and_sigma(xy_ph)
        
        # DRAW FROM q(z|x,y)
        qzx_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension, zx_mean, zx_log_sig_sq)
        
        # GET r(x|z,y)
        qzx_samp_y = tf.concat([qzx_samp,y_ph_n],1)
        reconstruction_xzx = autoencoder.calc_reconstruction(qzx_samp_y)
        x_mean_vae = reconstruction_xzx[0]
        x_log_sig_sq_vae = reconstruction_xzx[1]
        
        # COST FROM RECONSTRUCTION
        normalising_factor_x_vae = - 0.5 * tf.log(SMALL_CONSTANT+tf.exp(x_log_sig_sq_vae)) - 0.5 * np.log(2 * np.pi)
        square_diff_between_mu_and_x_vae = tf.square(x_mean_vae - x_ph_n)
        inside_exp_x_vae = -0.5 * tf.div(square_diff_between_mu_and_x_vae,SMALL_CONSTANT+tf.exp(x_log_sig_sq_vae))
        reconstr_loss_x_vae = -tf.reduce_sum(normalising_factor_x_vae + inside_exp_x_vae, 1)
        cost_R_vae = tf.reduce_mean(reconstr_loss_x_vae)
        
        # KL(q(z|x,y)||r(z|y))
        v_mean = zy_mean #2
        aux_mean = zx_mean #1
        v_log_sig_sq = tf.log(tf.exp(zy_log_sig_sq)+SMALL_CONSTANT) #2
        aux_log_sig_sq = tf.log(tf.exp(zx_log_sig_sq)+SMALL_CONSTANT) #1
        v_log_sig = tf.log(tf.sqrt(tf.exp(v_log_sig_sq))) #2
        aux_log_sig = tf.log(tf.sqrt(tf.exp(aux_log_sig_sq))) #1
        cost_VAE_a = v_log_sig-aux_log_sig+tf.divide(tf.exp(aux_log_sig_sq)+tf.square(aux_mean-v_mean),2*tf.exp(v_log_sig_sq))-0.5
        cost_VAE_b = tf.reduce_sum(cost_VAE_a,1)
        KL_vae = tf.reduce_mean(cost_VAE_b)
        
        # THE VICI COST FUNCTION
        lam_ph = tf.placeholder(dtype=tf.float32, name="lam_ph")
        COST_VAE = KL_vae+cost_R_vae
        
        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VICI")]
        
        # DRAW FROM q(x|y)
        #qx_samp = autoencoder_ENC._sample_from_gaussian_dist(bs_ph, xsh[1], x_mean, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq)))
        
        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(var_list_VICI)
        saver_VICI.restore(session,load_dir)
                
    ynt = y_data_h
    cost_value_vae, KL_VAE = session.run([COST_VAE, KL_vae], feed_dict={bs_ph:xsh[0], x_ph:x_data, y_ph:ynt, lam_ph:lam})
    ELBO = -cost_value_vae
    KL_DIV = KL_VAE
                
    return ELBO, KL_DIV
