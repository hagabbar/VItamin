################################################################################################################
#
# --Variational Inference for Computational Imaging (VICI) Multi-Fidelity Forward Model--
# 
# This model takes as input target images/objects and infers a simulated measured signal.
# There are two functions: 'train' and 'run'.
# 
# -train-
# takes as inputs a set of target images/objects, a set of associated low fidelity signal predictions
# (e.g. from simulations) and a set of high fidelity observations (e.g. experimental or better 
# simulations) and learns a variational inference model that predicts high fidelity observations from
# low fidelity ones and images/objects.
#
# INPUTS:
# params - list of parameters (see example for the necessary parameters).
# x_data - 2D array of training images/objects, where different samples are along dimension 0 and
#          values (e.g. pixel values) are along dimension 1.
# y_data_l - 2D array of training low fidelity observations, where different samples are along 
#            dimension 0 and values are along dimension 1. dimension 0 must be the same as that of
#             x_data (same number of samples as low fidelity observations).
# siz_high_res - dimensionality of high fidelity observations (may differ from the low fidelity ones).
# load_dir - directory from which to load the weights of the trained forward.
# save_dir - directory in which to save the weights of the trained model.
#        
# OUTPUTS: 
# COST_PLOT_MF - a vector containing the values of the cost function evaluated over a training subset
#                at intervals specified by the parameter 'report_interval_fw'.
# KL_PLOT_MF - a vector containing the values of the KL divergence evaluated over a training subset
#                at intervals specified by the parameter 'report_interval_fw'.
# 
#
# -run-
# takes as inputs a set of target images/objects and a set of associated low fidelity signal predictions
# (e.g. from simulations). It then loads weights previously trained using 'train' and returns predictions
# of the high fidelity observations.
# 
# INPUTS:
# params - list of parameters (see example for the necessary parameters).
# x_data_test - 2D array of images/objects, where different samples are along dimension 0 and
#               values (e.g. pixel values) are along dimension 1.
# y_data_test_l - 2D array of low fidelity observations, where different samples are along dimension
#                 0 and values are along dimension 1. dimension 0 must be the same as that of
#             x_data_test (same number of samples as low fidelity observations).
# siz_high_res - dimensionality of high fidelity observations (may differ from the low fidelity ones).
# load_dir - directory from which to load the weights of a trained model.
# 
# OUTPUTS:
# y_mean - empirical means of predicted observations
# y_std - empirical standard deviations of predicted observations    
#
#
#
################################################################################################################

import numpy as np
import tensorflow as tf
from sys import exit

from Neural_Networks import OELBO_decoder_difference
from Neural_Networks import OELBO_encoder
from Neural_Networks import VAE_encoder
from Neural_Networks import batch_manager
from Neural_Networks import vae_utils

def tf_normalise_dataset(xp):
     
    Xs = tf.shape(xp)
    
    l2norm = tf.sqrt(tf.reduce_sum(tf.multiply(xp,xp),1))
    l2normr = tf.reshape(l2norm,[Xs[0],1])
    x_data = tf.divide(xp,l2normr)
#    x_data = xp / l2norm.reshape(Xs[0],1)
   
    x_data=xp
    return x_data

def train(params, x_data, y_data_h, y_data_l, save_dir, plotter):
    
    # LOAD DATA
    x_data = x_data[0:np.shape(y_data_h)[0],:]
    y_data_train_h = y_data_h
    y_data_train_l = y_data_l[0:np.shape(y_data_h)[0],:]
    
    # USEFUL SIZES
    xsh = np.shape(x_data)
    ysh_h = np.shape(y_data_train_h)
    ysh_l = np.shape(y_data_train_l)
    
    z_dimension = params['z_dimensions_fw']
    n_weights = params['n_weights_fw']
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        
        # PLACE HOLDERS
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_ph")
        yl_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh_l[1]], name="yl_ph")
        yh_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh_h[1]], name="yl_ph")
        
        # LOAD NEURAL NETWORKS
        DEC_XYlZtoYh = OELBO_decoder_difference.VariationalAutoencoder("OELBO_decoder", ysh_h[1], z_dimension+ysh_l[1]+xsh[1], n_weights) # p(Yh|X,Yl,Z)
        ENC_XYltoZ = OELBO_encoder.VariationalAutoencoder("OELBO_encoder", ysh_l[1]+xsh[1], z_dimension, n_weights) # p(Z|X,Yl)
        ENC_XYhYltoZ = VAE_encoder.VariationalAutoencoder("vae_encoder", xsh[1]+ysh_l[1]+ysh_h[1], z_dimension, n_weights) # q(Z|X,Yl,Yh)
        
        # DEFINE MULTI-FIDELITY OBJECTIVE FUNCTION
        #####################################################################################################################
        SMALL_CONSTANT = 1e-6
        
        # NORMALISE INPUTS
        yl_ph_n = tf_normalise_dataset(yl_ph)
        yh_ph_n = tf_normalise_dataset(yh_ph)
        #yl_ph_n = yl_ph
        #yh_ph_n = yh_ph
#        yh_ph_d = tf_normalise_dataset(yh_ph-y_mu)
        yh_ph_d = yh_ph_n-yl_ph_n
        x_ph_n = tf_normalise_dataset(x_ph)
        #x_ph_n = x_ph
        
        # GET p(Z|X,Yl)
        zxyl_mean,zxyl_log_sig_sq = ENC_XYltoZ._calc_z_mean_and_sigma(tf.concat([x_ph_n,yl_ph_n],1))
        rxyl_samp = ENC_XYhYltoZ._sample_from_gaussian_dist(tf.shape(x_ph_n)[0], z_dimension, zxyl_mean, tf.log(tf.exp(zxyl_log_sig_sq)+SMALL_CONSTANT))
        
        # GET p(Yh|X,Yl,Z) FROM SAMPLES Z ~ p(Z|X,Yl)
        reconstruction_yh = DEC_XYlZtoYh.calc_reconstruction(tf.concat([x_ph_n,yl_ph_n,rxyl_samp],1))
        yh_diff = reconstruction_yh[0]
        yh_mean = yl_ph_n+yh_diff
        yh_log_sig_sq = reconstruction_yh[1]
#        yh_log_sig_sq = -10*tf.ones(tf.shape(reconstruction_yh[1]))
        
        # GET q(Z|X,Yl,Yh)
        zq_mean,zq_log_sig_sq = ENC_XYhYltoZ._calc_z_mean_and_sigma(tf.concat([x_ph_n,yl_ph_n,yh_ph_d],1))
        qzq_samp = ENC_XYhYltoZ._sample_from_gaussian_dist(tf.shape(x_ph_n)[0], z_dimension, zq_mean, tf.log(tf.exp(zq_log_sig_sq)+SMALL_CONSTANT))
        
        # GET p(Yh|X,Yl,Z) FROM SAMPLES Z ~ q(Z|X,Yl,Yh)
        reconstruction_yhq = DEC_XYlZtoYh.calc_reconstruction(tf.concat([x_ph_n,yl_ph_n,qzq_samp],1))
        yh_mean_vae = reconstruction_yhq[0]
        yh_log_sig_sq_vae = reconstruction_yhq[1]
#        yh_log_sig_sq_vae = -10*tf.ones(tf.shape(reconstruction_yh[1]))
        
        # COST FROM RECONSTRUCTION
        normalising_factor_yh_vae = - 0.5 * tf.log(SMALL_CONSTANT+tf.exp(yh_log_sig_sq_vae)) - 0.5 * np.log(2 * np.pi)
        square_diff_between_mu_and_yh_vae = tf.square(yh_mean_vae - yh_ph_d)
        inside_exp_yh_vae = -0.5 * tf.div(square_diff_between_mu_and_yh_vae,SMALL_CONSTANT+tf.exp(yh_log_sig_sq_vae))
        reconstr_loss_yh_vae = -tf.reduce_sum(normalising_factor_yh_vae + inside_exp_yh_vae, 1)
        cost_R_vae = tf.reduce_mean(reconstr_loss_yh_vae)
        
        # KL(q(Z|X,Yl,Yh)||p(Z|X,Yl))
        v_mean = zxyl_mean #2
        aux_mean = zq_mean #1
        v_log_sig_sq = tf.log(tf.exp(zxyl_log_sig_sq)+SMALL_CONSTANT) #2
        aux_log_sig_sq = tf.log(tf.exp(zq_log_sig_sq)+SMALL_CONSTANT) #1
        v_log_sig = tf.log(tf.sqrt(tf.exp(v_log_sig_sq))) #2
        aux_log_sig = tf.log(tf.sqrt(tf.exp(aux_log_sig_sq))) #1
        cost_VAE_a = v_log_sig-aux_log_sig+tf.divide(tf.exp(aux_log_sig_sq)+tf.square(aux_mean-v_mean),2*tf.exp(v_log_sig_sq))-0.5
        cost_VAE_b = tf.reduce_sum(cost_VAE_a,1)
        KL_vae = tf.reduce_mean(cost_VAE_b)
        
        # THE VICI COST FUNCTION
        COST = KL_vae+cost_R_vae
        
        ######################################################################################################################
        
        # TRAIN MULTI-FIDELITY MODEL
        var_list_ELBO = [var for var in tf.trainable_variables() if var.name.startswith("ELBO")]
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate_fw']) 
        minimize = optimizer.minimize(COST,var_list = var_list_ELBO)
        
        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        session.run(init)
    
    # OPTIMISATION OF MULTI-FIDELITY MODEL
    ##################################################################################################################################################
    COST_PLOT_MF = np.zeros(np.int(np.round(params['num_iterations_fw']/params['report_interval_fw'])+1))
    KL_PLOT_MF = np.zeros(np.int(np.round(params['num_iterations_fw']/params['report_interval_fw'])+1))
    print('Training Multi-Fidelity Model...')
    indices_generator = batch_manager.SequentialIndexer(params['batch_size_fw'], xsh[0])
    nif = -1
    for i in range(params['num_iterations_fw']):
        next_indices = indices_generator.next_indices()
        session.run(minimize, feed_dict={x_ph:x_data[next_indices, :], yl_ph:y_data_train_l[next_indices, :], yh_ph:y_data_train_h[next_indices, :]})
        if i % params['report_interval_fw'] == 0:
            nif = nif+1
            COST_MF, KL_div = session.run([COST,KL_vae], feed_dict={x_ph:x_data[0:100, :], yl_ph:y_data_train_l[0:100, :], yh_ph:y_data_train_h[0:100, :]})
            COST_PLOT_MF[nif] = COST_MF
            KL_PLOT_MF[nif] = KL_div

            if params['print_values']==True:
                print('--------------------------------------------------------------')
                print('Iteration:',i)
                print('Cost Multi-Fidelity Model:',COST_MF)
                print('Multi-Fidelity Model KL Divergence:',KL_div)
    
        if i % params['save_interval_fw'] == 0:
            save_path = saver.save(session,save_dir)

            # Generate forward estimation results
            plotter.plot_y_test(i)
            plotter.plot_y_dist(i)
    
    return COST_PLOT_MF, KL_PLOT_MF
    
def run(params, x_data_test, y_data_test_l, siz_high_res, load_dir):
    
    # USEFUL SIZES
    xsh = np.shape(x_data_test)
    ysh1_h = siz_high_res
    ysh_l = np.shape(y_data_test_l)
    
    z_dimension = params['z_dimensions_fw']
    n_weights = params['n_weights_fw']
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        
        # PLACE HOLDERS
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_ph")
        yl_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh_l[1]], name="yl_ph")
        
        # LOAD NEURAL NETWORKS
        DEC_XYlZtoYh = OELBO_decoder_difference.VariationalAutoencoder("OELBO_decoder", ysh1_h, z_dimension+ysh_l[1]+xsh[1], n_weights) # p(Yh|X,Yl,Z)
        ENC_XYltoZ = OELBO_encoder.VariationalAutoencoder("OELBO_encoder", ysh_l[1]+xsh[1], z_dimension, n_weights) # p(Z|X,Yl)
        ENC_XYhYltoZ = VAE_encoder.VariationalAutoencoder("vae_encoder", xsh[1]+ysh_l[1]+ysh1_h, z_dimension, n_weights) # q(Z|X,Yl,Yh)
        
        # DEFINE MULTI-FIDELITY OBJECTIVE FUNCTION
        #####################################################################################################################
        SMALL_CONSTANT = 1e-6
        
        # NORMALISE INPUTS
        yl_ph_n = tf_normalise_dataset(yl_ph)
        #yl_ph_n = yl_ph
#        yh_ph_d = tf_normalise_dataset(yh_ph-y_mu)
        x_ph_n = tf_normalise_dataset(x_ph)
        #x_ph_n = x_ph        

        # GET p(Z|X,Yl)
        zxyl_mean,zxyl_log_sig_sq = ENC_XYltoZ._calc_z_mean_and_sigma(tf.concat([x_ph_n,yl_ph_n],1))
        rxyl_samp = ENC_XYhYltoZ._sample_from_gaussian_dist(tf.shape(x_ph_n)[0], z_dimension, zxyl_mean, tf.log(tf.exp(zxyl_log_sig_sq)+SMALL_CONSTANT))
        
        # GET p(Yh|X,Yl,Z) FROM SAMPLES Z ~ p(Z|X,Yl)
        reconstruction_yh = DEC_XYlZtoYh.calc_reconstruction(tf.concat([x_ph_n,yl_ph_n,rxyl_samp],1))
        yh_diff = reconstruction_yh[0]
        yh_mean = yl_ph_n+yh_diff
        yh_log_sig_sq = reconstruction_yh[1]
        yh_samp = ENC_XYhYltoZ._sample_from_gaussian_dist(tf.shape(x_ph_n)[0], tf.shape(yh_mean)[1], yh_mean, tf.log(tf.exp(yh_log_sig_sq)+SMALL_CONSTANT))
#        yh_log_sig_sq = -10*tf.ones(tf.shape(reconstruction_yh[1]))
        
        ######################################################################################################################
        
        # INITIALISE AND RUN SESSION
        var_list_ELBO = [var for var in tf.trainable_variables() if var.name.startswith("ELBO")]
        init = tf.initialize_all_variables()
        session.run(init)
        saver_ELBO = tf.train.Saver(var_list_ELBO)
        saver_ELBO.restore(session,load_dir)
    
    # RUN MULTIFIDELITY INFERENCE
    ##################################################################################################################################################
    
    y_multi, y_d, yl_n, y_ds = session.run([yh_mean,yh_diff,yl_ph_n,yh_log_sig_sq], feed_dict={x_ph:x_data_test, yl_ph:y_data_test_l})
            
    YS = np.zeros((np.shape(y_multi)[0],np.shape(y_multi)[1],100))
    for ir in range(100):
        y_samp = session.run(yh_mean, feed_dict={x_ph:x_data_test, yl_ph:y_data_test_l})
        YS[:,:,ir] = y_samp
     
    y_samp = session.run(yh_samp, feed_dict={x_ph:x_data_test, yl_ph:y_data_test_l})
    y_mean = np.mean(YS,axis=2)
    y_std = np.std(YS,axis=2)
    
    return y_mean, y_std, y_samp

    ###########################################################################################################################################################################
