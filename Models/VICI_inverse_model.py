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
import tensorflow as tf

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

# NORMALISE DATASET FUNCTION
def tf_normalise_dataset(xp):
    
    Xs = tf.shape(xp)
    
    l2norm = tf.sqrt(tf.reduce_sum(tf.multiply(xp,xp),1))
    l2normr = tf.reshape(l2norm,[Xs[0],1])
    x_data = tf.divide(xp,l2normr)
  
    # comment this if you want to use normalise 
    x_data = xp 
    return x_data

#def train(params, load_dir, save_dir, plotter, y_data_test, pos_test, samples):
def train(params, x_data, y_data, siz_high_res, save_dir, plotter, y_data_test,train_files,normscales,y_data_train_noisefree,y_normscale):    

    x_data = x_data
    y_data_train_l = y_data

    # USEFUL SIZES
    xsh = np.shape(x_data)
    yshl1 = np.shape(y_data)[1]
    ysh1 = np.shape(y_data)[1]    
    print(xsh,yshl1,ysh1)
 
    #z_dimension_fm = params['z_dimensions_fw']
    #n_weights_fm = params['n_weights_fw']
    
    z_dimension = params['z_dimension']
    bs = params['batch_size']
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
        yt_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="yt_ph")
        
        # LOAD FORWARD MODEL NEURAL NETWORKS
        #DEC_XYlZtoYh = OELBO_decoder_difference.VariationalAutoencoder("OELBO_decoder", ysh1, z_dimension_fm+yshl1+xsh[1], n_weights_fm) # p(Yh|X,Yl,Z)
        #ENC_XYltoZ = OELBO_encoder.VariationalAutoencoder("OELBO_encoder", yshl1+xsh[1], z_dimension_fm, n_weights_fm) # p(Z|X,Yl)
        #ENC_XYhYltoZ = VAE_encoder.VariationalAutoencoder("vae_encoder", xsh[1]+ysh1+yshl1, z_dimension_fm, n_weights_fm) # q(Z|X,Yl,Yh)
        
        # LOAD VICI NEURAL NETWORKS
        autoencoder = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh[1], z_dimension+ysh1, n_weights) # r(x|z,y)
        autoencoder_ENC = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights) # generates params for r(z|y)
        autoencoder_VAE = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh[1]+ysh1, z_dimension, n_weights) # used to sample from r(z|y)?
        
        # DEFINE MULTI-FIDELITY FORWARD MODEL
        #####################################################################################################################
        SMALL_CONSTANT = 1e-6
        
        # NORMALISE INPUTS
        yl_ph_n = tf_normalise_dataset(yt_ph) # placeholder for normalised low-res y data
        x_ph_n = tf_normalise_dataset(x_ph)   # placeholder for normalised x data
        #yl_ph_n = yt_ph
        #x_ph_n = x_ph
        #yl_ph_n = tf.Print(yl_ph_n, [yl_ph_n], first_n=1, summarize=10, message="Thss is yl_ph_n: ")
        #x_ph_n = tf.Print(x_ph_n, [x_ph_n], first_n=1, summarize=10, message="This is x_ph_n: ")

        # GET p(Z|X,Yl) - takes in x data and low res y data and returns mean and logvar of Gaussian z distribution
        #zxyl_mean,zxyl_log_sig_sq = ENC_XYltoZ._calc_z_mean_and_sigma(tf.concat([x_ph_n,yl_ph_n],1))
        #zxyl_mean = tf.Print(zxyl_mean, [zxyl_mean], first_n=1, summarize=10, message="Thss is zxyl_mean: ")
        #zxyl_log_sig_sq = tf.Print(zxyl_log_sig_sq, [zxyl_log_sig_sq], first_n=1, summarize=10, message="Thss is zxyl_log_sig_sq: ")
        # then samples z from that distribution
        #rxyl_samp = ENC_XYhYltoZ._sample_from_gaussian_dist(tf.shape(x_ph_n)[0], z_dimension_fm, zxyl_mean, tf.log(tf.exp(zxyl_log_sig_sq)+SMALL_CONSTANT))
        #rxyl_samp = tf.Print(rxyl_samp, [rxyl_samp], first_n=1, summarize=10, message="Thss is rxyl_samp: ")        

        # GET p(Yh|X,Yl,Z) FROM SAMPLES Z ~ p(Z|X,Yl)
        # then decodes back to high res y data 
        #reconstruction_yh = DEC_XYlZtoYh.calc_reconstruction(tf.concat([x_ph_n,yl_ph_n,rxyl_samp],1))
        # = tf.Print(, [], first_n=1, summarize=10, message="Thss is : ")
        #reconstruction_yh = tf.Print(reconstruction_yh, [reconstruction_yh], first_n=1, summarize=10, message="Thss is reconstruction_yh: ")
        #yh_diff = reconstruction_yh[0] # looks like the difference between the low res y data and the mean reconstructed high res y data
        #yh_diff = tf.Print(yh_diff, [yh_diff], first_n=1, summarize=10, message="Thss is yh_diff: ") 
        #yh_mean = yl_ph_n+yh_diff      # the mean reconstrcted high res data
        #yh_mean = tf.Print(yh_mean, [yh_mean], first_n=1, summarize=10, message="Thss is yh_mean: ")
        #yh_log_sig_sq = reconstruction_yh[1]  # the reconstructed high res y data logvar
        #yh_log_sig_sq = tf.Print(yh_log_sig_sq, [yh_log_sig_sq], first_n=1, summarize=10, message="Thss is yh_log_sig_sq: ")
        # then sample something? y doesn't seem to be used anywhere after this 
        # this ends up being the reconstruction of the exact y data corresponding to each x data input
        #y = ENC_XYhYltoZ._sample_from_gaussian_dist(tf.shape(yt_ph)[0], tf.shape(yt_ph)[1], yh_mean, yh_log_sig_sq)
        #y = tf.Print(y, [y], first_n=1, summarize=10, message="Thss is y: ")        

        # DRAW SYNTHETIC Ys TRAINING DATA
#        _, y = OM.forward_model(x_ph_amp,x_ph_ph)
        
        ##########################################################################################################################################
        
        # GET r(z|y)
        #y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph")  # placeholder for y data
        #y_ph_n = tf_normalise_dataset(y_ph)                                       # placeholder for normalised y data
        #y_ph = tf.Print(y_ph, [y_ph], first_n=1, summarize=10, message="Thss is y_ph: ")
        #y_ph_n = y_ph
        # run inverse autoencoder to generate mean and logvar of z given y data - these are the parameters for r(z|y)
        #zy_mean,zy_log_sig_sq = autoencoder_ENC._calc_z_mean_and_sigma(y_ph_n) 
        zy_mean,zy_log_sig_sq = autoencoder_ENC._calc_z_mean_and_sigma(yl_ph_n)        

        # DRAW FROM r(z|y) - given the Gaussian parameters generate z samples
        rzy_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension, zy_mean, zy_log_sig_sq)
        
        # GET r(x|z,y) from r(z|y) samples
        #rzy_samp_y = tf.concat([rzy_samp,y_ph_n],1)
        rzy_samp_y = tf.concat([rzy_samp,yl_ph_n],1)
        reconstruction_xzy = autoencoder.calc_reconstruction(rzy_samp_y)
        x_mean = reconstruction_xzy[0]
        x_log_sig_sq = reconstruction_xzy[1]
        
        # KL(r(z|y)||p(z))
        #latent_loss = -0.5 * tf.reduce_sum(1 + zy_log_sig_sq - tf.square(zy_mean) - tf.exp(zy_log_sig_sq), 1)
        #KL = tf.reduce_mean(latent_loss)
       
        # GET q(z|x,y)
        #xy_ph = tf.concat([x_ph_n,y_ph_n],1)
        xy_ph = tf.concat([x_ph_n,yl_ph_n],1)
        zx_mean,zx_log_sig_sq = autoencoder_VAE._calc_z_mean_and_sigma(xy_ph)
 
        # DRAW FROM q(z|x,y)
        qzx_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension, zx_mean, zx_log_sig_sq)
        
        # GET r(x|z,y)
        #qzx_samp_y = tf.concat([qzx_samp,y_ph_n],1)
        qzx_samp_y = tf.concat([qzx_samp,yl_ph_n],1)
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
        KL_vae = tf.reduce_mean(cost_VAE_b)                               # computes the mean over all tensor elements
        
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
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        session.run(init)
        #saver_ELBO = tf.train.Saver(var_list_ELBO)
        #saver_ELBO.restore(session,load_dir)
        saver = tf.train.Saver()
    
    KL_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1)) # vector to store test OELBO values
    COST_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1)) # vector to store test VAE ELBO values

    print('Training Inference Model...')    
    # START OPTIMISATION OF OELBO
    indices_generator = batch_manager.SequentialIndexer(params['batch_size'], xsh[0])
    ni = -1
    test_n = 100
    olvec = []
    for i in range(params['num_iterations']):
        if params['do_load_in_chunks']:
            # Get next set of training samples every 1000 iterations
            if int(i) >= 1000 and int(i) % 1000 == 0:
                x_data_train, y_data_train_l, y_data_train_h, x_data_train_h, y_data_train_lh =  chris_data.load_training_set(params,train_files,normscales)

                x_data = x_data_train
                y_data_train_l = y_data_train_l
                

        next_indices = indices_generator.next_indices()
        
        # run the session - input batchsize the x-data training batch and the y-data training batch 
        #yn = session.run(y, feed_dict={bs_ph:bs, x_ph:x_data[next_indices, :], yt_ph:y_data_train_l[next_indices, :]})
        #session.run(minimize, feed_dict={bs_ph:bs, x_ph:x_data[next_indices, :],  y_ph:yn, lam_ph:lam, yt_ph:y_data_train_l[next_indices, :]}) # minimising cost function

        # Make 25 noise realizations
        if params['do_extra_noise']:
            x_data_train_l = x_data[next_indices,:]
            y_data_train_l = y_data_train_noisefree[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],params['ndata']))
            y_data_train_l /= y_normscale[0]
            #print('generated {} elements of new training data noise'.format(params['batch_size']))

            session.run(minimize, feed_dict={bs_ph:bs, x_ph:x_data_train_l, lam_ph:lam, yt_ph:y_data_train_l}) # minimising cost function
        else:
            session.run(minimize, feed_dict={bs_ph:bs, x_ph:x_data[next_indices, :], lam_ph:lam, yt_ph:y_data_train_l[next_indices, :]}) # minimising cost function

        if i % params['report_interval'] == 0 and i > 0:
            ni = ni+1
                
            #ynt = session.run(y, feed_dict={bs_ph:test_n, x_ph:x_data[0:test_n,:], yt_ph:y_data_train_l[0:test_n,:]})
            #cost_value_vae, KL_VAE = session.run([COST_VAE, KL_vae], feed_dict={bs_ph:test_n, x_ph:x_data[0:test_n,:], y_ph:ynt, lam_ph:lam, yt_ph:y_data_train_l[0:test_n,:]})
            cost_value_vae, KL_VAE = session.run([COST_VAE, KL_vae], feed_dict={bs_ph:test_n, x_ph:x_data[0:test_n,:], lam_ph:lam, yt_ph:y_data_train_l[0:test_n,:]})
            KL_PLOT[ni] = KL_VAE
            COST_PLOT[ni] = cost_value_vae

            # make log loss plot
            fig_loss, axes_loss = plt.subplots(1,figsize=(10,8))
            axes_loss.grid()
            axes_loss.set_ylabel('Loss')
            axes_loss.set_xlabel('Iterations elapsed: %s' % i)
            axes_loss.semilogy(np.arange(len(KL_PLOT)), np.abs(KL_PLOT), label='KL')
            axes_loss.semilogy(np.arange(len(COST_PLOT)), np.abs(COST_PLOT), label='COST')
            axes_loss.legend(loc='upper left')
            plt.savefig('%s/latest/losses_logscale.png' % params['plot_dir'])
            plt.close(fig_loss)

            # make non-log scale loss plot
            fig_loss, axes_loss = plt.subplots(1,figsize=(10,8))
            axes_loss.grid()
            axes_loss.set_ylabel('Loss')
            axes_loss.set_xlabel('Iterations elapsed: %s' % i)
            axes_loss.plot(np.arange(len(KL_PLOT)), KL_PLOT, label='KL')
            axes_loss.plot(np.arange(len(COST_PLOT)), COST_PLOT, label='COST')
            axes_loss.set_xscale('log')
            axes_loss.set_yscale('log')
            axes_loss.legend(loc='upper left')
            plt.savefig('%s/latest/losses.png' % params['plot_dir'])
            plt.close(fig_loss)
                
            if params['print_values']==True:
                print('--------------------------------------------------------------')
                print('Iteration:',i)
                print('Training Set -ELBO:',cost_value_vae)
                print('KL Divergence:',KL_VAE)
       
        if i % params['save_interval'] == 0 and i > 0:

            # Save model 
            save_path = saver.save(session,save_dir)
                
                
    return COST_PLOT, KL_PLOT, train_files 

#def resume_training(params, x_data, y_data_l, siz_high_res, load_dir, save_dir):
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
        
        if params['do_load_in_chunks']:
            # Get next set of training samples every 1000 iterations
            if int(i) >= 1000 and int(i) % 1000 == 0:
                print('Getting training new set ...')
                x_data_train, y_data_train_l, y_data_train_h, x_data_train_h, y_data_train_lh =  chris_data.load_training_set(params,train_files,normscales)

                x_data = x_data_train
                y_data_train_l = y_data_train_l
        

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
                COST_PLOT[ni] = cost_value_vae
                
                if params['print_values']==True:
                    print('--------------------------------------------------------------')
                    print('Iteration:',i)
                    print('Training Set ELBO:',-cost_value_vae)
                    print('KL Divergence:',KL_VAE)
       
        if i % params['save_interval'] == 0:
             
            save_path = saver.save(session,save_dir)
                
                
    return COST_PLOT, KL_PLOT, train_files

def run(params, y_data_test, siz_x_data, load_dir):

    # USEFUL SIZES
    xsh1 = siz_x_data
    ysh1 = np.shape(y_data_test)[1]
    
    z_dimension = params['z_dimension']
    n_weights = params['n_weights']
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-6
        
        # LOAD VICI NEURAL NETWORKS
        autoencoder = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh1, z_dimension+ysh1, n_weights)
        autoencoder_ENC = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights)
        autoencoder_VAE = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh1+ysh1, z_dimension, n_weights)
        
        # GET r(z|y)
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph")
        y_ph_n = tf_normalise_dataset(y_ph)
        #y_ph_n = y_ph
        zy_mean,zy_log_sig_sq = autoencoder_ENC._calc_z_mean_and_sigma(y_ph_n)
        
        # DRAW FROM r(z|y)
        rzy_samp = autoencoder_VAE._sample_from_gaussian_dist(tf.shape(y_ph_n)[0], z_dimension, zy_mean, zy_log_sig_sq)
        
        # GET r(x|z,y) from r(z|y) samples
        rzy_samp_y = tf.concat([rzy_samp,y_ph_n],1)
        reconstruction_xzy = autoencoder.calc_reconstruction(rzy_samp_y)
        x_mean = reconstruction_xzy[0]
        x_log_sig_sq = reconstruction_xzy[1]
        
        # GET pseudo max
        rzy_samp_y_pm = tf.concat([zy_mean,y_ph_n],1)
        reconstruction_xzy_pm = autoencoder.calc_reconstruction(rzy_samp_y_pm)
        x_pmax = reconstruction_xzy_pm[0]
        
        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VICI")]
        
        # DRAW FROM q(x|y)
        qx_samp = autoencoder_ENC._sample_from_gaussian_dist(tf.shape(y_ph_n)[0], xsh1, x_mean, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq)))
        
        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(var_list_VICI)
        saver_VICI.restore(session,load_dir)
    
    # ESTIMATE TEST SET RECONSTRUCTION PER-PIXEL APPROXIMATE MARGINAL LIKELIHOOD and draw from q(x|y)
    n_ex_s = params['n_samples'] # number of samples to save per reconstruction
    ns = np.maximum(100,n_ex_s) # number of samples to use to estimate per-pixel marginal
    
    XM = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    XSX = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    XSA = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
   
    if params['do_m1_m2_cut']: 
        for i in range(ns):
            counter = False
            cnt = 0
            while counter == False:
                rec_x_m = session.run(x_mean,feed_dict={y_ph:y_data_test})
                rec_x_mx = session.run(qx_samp,feed_dict={y_ph:y_data_test})
                rec_x_s = session.run(x_mean,feed_dict={y_ph:y_data_test})
                if (rec_x_mx[cnt,0] <= 35.0) or (rec_x_mx[cnt,0] >= 80.0) or (rec_x_mx[cnt,2] <= 35.0) or (rec_x_mx[cnt,2] >= 80.0):
                    continue
                else:
                    XSX[cnt,:,i] = rec_x_mx
                    print('Predictions generated for test sampe %s and parameter %s ...' % (str(cnt),str(i)))
                    cnt+=1
                if cnt == np.shape(y_data_test)[0]:
                    counter = True

    else:
        for i in range(ns):
            rec_x_m = session.run(x_mean,feed_dict={y_ph:y_data_test})
            rec_x_mx = session.run(qx_samp,feed_dict={y_ph:y_data_test})
            rec_x_s = session.run(x_mean,feed_dict={y_ph:y_data_test})

            XM[:,:,i] = rec_x_m
            XSX[:,:,i] = rec_x_mx
            XSA[:,:,i] = rec_x_s

    
    pmax = session.run(x_pmax,feed_dict={y_ph:y_data_test})
    
    xm = np.mean(XM,axis=2)
    xsx = np.std(XSX,axis=2)
    xs = np.std(XM,axis=2)
    XS = XSX[:,:,0:n_ex_s]
    #XS = XSA[:,:,0:n_ex_s]

    """
    n_steps=0
    i = False
    while i == False:
        rec_x_m = session.run(x_mean,feed_dict={y_ph:y_data_test})
        rec_x_mx = session.run(qx_samp,feed_dict={y_ph:y_data_test})
        rec_x_s = session.run(x_mean,feed_dict={y_ph:y_data_test})
        # remove m1 and m2 samples outside of prior
        cont = False
        for j in range(XSX.shape[0]):
            if (rec_x_mx[j,0] >= 35.0) and (rec_x_mx[j,0] <= 50.0) and (rec_x_mx[j,2] <= 50.0) and (rec_x_mx[j,2] >= 35.0):
                continue
            else: cont=True
        if cont==True:
            continue
        elif n_steps==ns:
            i=True
        else:
            XM[:,:,i] = rec_x_m
            XSX[:,:,i] = rec_x_mx
            XSA[:,:,i] = rec_x_s
            ns+=1
    """
    # remove m1 and m2 samples outside of prior
    """
    new_XS = []
    #print(XS[0,:,:].shape) test samples, parameter, posterior
    max_len=n_ex_s
    for j in range(XSX.shape[0]):
        mask = [(XSX[j,0,:] >= 35.0) & (XSX[j,0,:] <= 50.0) & (XSX[j,2,:] <= 50.0) & (XSX[j,2,:] >= 35.0)]
        mask = np.argwhere(mask[0])
        temp_XS = XSX[j,:,mask]
        temp_XS = temp_XS.reshape(XSX[j,:,mask].shape[2],XSX[j,:,mask].shape[0])
        new_XS.append(temp_XS[:,0:max_len])
    new_XS = np.array(new_XS)
    print(new_XS.shape)
    exit()
    XS = XSX[:,:,0:n_ex_s]
    """
    #XS = XSA[:,:,0:n_ex_s]
    
                
    return xm, xsx, XS, pmax

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
        qx_samp = autoencoder_ENC._sample_from_gaussian_dist(bs_ph, xsh[1], x_mean, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq)))
        
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
