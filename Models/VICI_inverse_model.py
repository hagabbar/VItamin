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

def train(params, x_data, y_data, x_data_test, y_data_test, y_normscale, save_dir):    

    # USEFUL SIZES
    xsh = np.shape(x_data)
    ysh = np.shape(y_data)[1]
    z_dimension = params['z_dimension']
    bs = params['batch_size']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']
    nKL = params['KL_cycles']    

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        
        # PLACE HOLDERS
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_ph")
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph") # batch size placeholder
        yt_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh], name="yt_ph")
        eps = tf.placeholder(dtype=tf.float32, shape=[None, z_dimension, nKL], name="eps")        

        # LOAD VICI NEURAL NETWORKS
        autoencoder = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh[1], z_dimension+ysh, n_weights_r2) # r2(x|z,y)
        autoencoder_ENC = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh, z_dimension, n_weights_r1) # generates params for r1(z|y)
        autoencoder_VAE = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh[1]+ysh, z_dimension, n_weights_q) # used to sample from q(z|x,y)?
        
        #tf.set_random_seed(42)
        #tf.set_random_seed(42)
        #eps = tf.random.stateless_normal(shape=[[bs_ph, z_dimension]], seed=42, mean=0, stddev=1., dtype=tf.float32)

        tf.set_random_seed(np.random.randint(0,10))
  
        # DEFINE MULTI-FIDELITY FORWARD MODEL
        #####################################################################################################################
        SMALL_CONSTANT = 1e-6
        
        # NORMALISE INPUTS
        #y_ph_n = tf_normalise_dataset(yt_ph) # placeholder for normalised low-res y data
        #x_ph_n = tf_normalise_dataset(x_ph)   # placeholder for normalised x data
        x_ph_n = x_ph
        y_ph_n = yt_ph

        ##########################################################################################################################################
        
        # GET r(z|y)
        # run inverse autoencoder to generate mean and logvar of z given y data - these are the parameters for r(z|y)
        zy_mean_a, zy_log_sig_sq_a, zy_mean_b, zy_log_sig_sq_b, ab = autoencoder_ENC._calc_z_mean_and_sigma(y_ph_n)        

        # DRAW FROM r(z|y) - given the Gaussian parameters generate z samples
        rzy_samp = autoencoder_ENC._sample_from_gaussian_dist(bs_ph, z_dimension, zy_mean_a, zy_log_sig_sq_a, zy_mean_b, zy_log_sig_sq_b, ab)
        
        # GET r(x|z,y) from r(z|y) samples
        rzy_samp_y = tf.concat([rzy_samp,y_ph_n],1)
        reconstruction_xzy = autoencoder.calc_reconstruction(rzy_samp_y)
        x_mean = reconstruction_xzy[0]
        x_log_sig_sq = reconstruction_xzy[1]
       
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
        inside_exp_x_vae = -0.5 * tf.divide(square_diff_between_mu_and_x_vae,SMALL_CONSTANT+tf.exp(x_log_sig_sq_vae))
        reconstr_loss_x_vae = tf.reduce_sum(normalising_factor_x_vae + inside_exp_x_vae, 1)
        cost_R_vae = -1.0*tf.reduce_mean(reconstr_loss_x_vae)
        
        # KL(q(z|x,y)||r(z|y))
        v_mean = zy_mean_a # means of r1
        aux_mean = zx_mean # means of q
        v_log_sig_sq = tf.log(tf.exp(zy_log_sig_sq_a)+SMALL_CONSTANT) # log variances of r1
        aux_log_sig_sq = tf.log(tf.exp(zx_log_sig_sq)+SMALL_CONSTANT) # log variances of q
        v_log_sig = tf.log(tf.sqrt(tf.exp(v_log_sig_sq))) # log stdevs of r1
        aux_log_sig = tf.log(tf.sqrt(tf.exp(aux_log_sig_sq))) # log stdevs of q
        cost_VAE_a = v_log_sig - aux_log_sig + tf.divide(tf.exp(aux_log_sig_sq)+tf.square(aux_mean-v_mean), 2*tf.exp(v_log_sig_sq)) - 0.5
        cost_VAE_b = tf.reduce_sum(cost_VAE_a,1)                           
        KL_vae = tf.reduce_mean(cost_VAE_b)                               # computes the mean over all tensor elements

        # compute quick montecarlo KL
        #quick_KL = 0.0
        #normalising_factor_r = - 0.5 * tf.log(SMALL_CONSTANT+tf.exp(zy_log_sig_sq)) - 0.5*tf.log(2.0*np.pi)
        analytic_ent_q = -tf.log(tf.sqrt((SMALL_CONSTANT+tf.exp(zx_log_sig_sq))*2.0*np.pi*tf.exp(1.0)))
        #for i in range(nKL):
            #new_qzx_samp = tf.add(zx_mean, tf.multiply(tf.sqrt(tf.exp(zx_log_sig_sq)), eps[:,:,i]))
        new_qzx_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension, zx_mean, zx_log_sig_sq)
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=ab),
                          components_distribution=tfd.MultivariateNormalDiag(
                              loc=tf.stack([zy_mean_a,zy_mean_b],axis=1),
                              scale_diag=tf.stack([tf.sqrt(tf.exp(zy_log_sig_sq_a)),tf.sqrt(tf.exp(zy_log_sig_sq_b))],axis=1)))
        temp = bimix_gauss.log_prob(new_qzx_samp)
            #normalising_factor_q = - 0.5 * tf.log(SMALL_CONSTANT+tf.exp(zx_log_sig_sq))
            #normalising_factor_r = - 0.5 * tf.log(SMALL_CONSTANT+tf.exp(zy_log_sig_sq)) - 0.5*tf.log(2.0*np.pi)
            #square_diff_between_q_and_q = tf.square(zx_mean - new_qzx_samp)         
            #square_diff_between_ra_and_q = tf.square(zy_mean - new_qzx_samp)
            #square_diff_between_rb_and_q = tf.square(zy_mean_b - new_qzx_samp)
            #inside_exp_q = -0.5 * tf.divide(square_diff_between_q_and_q,SMALL_CONSTANT+tf.exp(zx_log_sig_sq))
            #inside_exp_ra = -0.5 * tf.divide(square_diff_between_ra_and_q,SMALL_CONSTANT+tf.exp(zy_log_sig_sq))
            #inside_exp_rb = -0.5 * tf.divide(square_diff_between_rb_and_q,SMALL_CONSTANT+tf.exp(zy_log_sig_sq_b))
            #temp = ab*tf.exp() + tf.exp()
            #quick_KL_batch = tf.reduce_sum(normalising_factor_q - normalising_factor_r + inside_exp_q - inside_exp_r, 1)
            #analytic_ent_q = -tf.log(tf.sqrt((SMALL_CONSTANT+tf.exp(zx_log_sig_sq))*2.0*np.pi*tf.exp(1.0)))
        #temp = bimix_gauss.log_prob([new_qzx_samp])
        quick_KL_batch = tf.reduce_sum(analytic_ent_q, 1)
        quick_KL = tf.reduce_mean(quick_KL_batch - temp)
        #quick_KL /= tf.dtypes.cast(eps.shape[2],dtype=tf.float32),

        # THE VICI COST FUNCTION
        #lam_ph = tf.placeholder(dtype=tf.float32, name="lam_ph")
        COST_VAE = KL_vae + cost_R_vae
        COST = COST_VAE
        quick_COST_VAE = quick_KL + cost_R_vae        
        quick_COST = quick_COST_VAE

        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VICI")]
        #var_list_ELBO = [var for var in tf.trainable_variables() if var.name.startswith("ELBO")]
        
        # DEFINE OPTIMISER (using ADAM here)
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate']) 
        minimize = optimizer.minimize(COST,var_list = var_list_VICI)
        quick_minimize = optimizer.minimize(quick_COST,var_list = var_list_VICI)

        # DRAW FROM q(x|y)
        #qx_samp = autoencoder_ENC._sample_from_gaussian_dist(bs_ph, xsh[1], x_mean, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq)))
        
        # INITIALISE AND RUN SESSION
#        init = tf.variables_initializer(var_list_VICI)
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        session.run(init)
        #saver_ELBO = tf.train.Saver(var_list_ELBO)
        #saver_ELBO.restore(session,load_dir)
        saver = tf.train.Saver()
    
    #KL_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1)) # vector to store test OELBO values
    #COST_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1)) # vector to store test VAE ELBO values

    print('Training Inference Model...')    
    # START OPTIMISATION OF OELBO
    indices_generator = batch_manager.SequentialIndexer(params['batch_size'], xsh[0])
    #ni = -1
    #test_n = 100
    #olvec = []
    plotdata = []
    kldata = []
    eps_data = np.random.normal(0,1,(params['batch_size'],params['z_dimension'],params['KL_cycles']))
    for i in range(params['num_iterations']):

        next_indices = indices_generator.next_indices()
 
        # Make noise realizations and add to training data
        next_x_data = x_data[next_indices,:]
        next_y_data = y_data[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],params['ndata']))
        next_y_data /= y_normscale        

        # train to minimise the cost function
        #session.run(minimize, feed_dict={bs_ph:bs, x_ph:next_x_data, yt_ph:next_y_data, eps:eps_data})
        session.run(quick_minimize, feed_dict={bs_ph:bs, x_ph:next_x_data, yt_ph:next_y_data, eps:eps_data})

        # if we are in a report iteration extract cost function values
        if i % params['report_interval'] == 0 and i > 0:
            #ni = ni+1
                
            #cost_value_vae, KL_VAE = session.run([cost_R_vae, KL_vae], feed_dict={bs_ph:bs, x_ph:next_x_data, yt_ph:next_y_data})
            cost_value_vae, KL_VAE, old_KL, AB = session.run([cost_R_vae, quick_KL, KL_vae, ab], feed_dict={bs_ph:bs, x_ph:next_x_data, yt_ph:next_y_data, eps:eps_data})
            plotdata.append([cost_value_vae,KL_VAE,cost_value_vae+KL_VAE])
            kldata.append([old_KL, KL_VAE])
            
            #KL_PLOT[ni] = KL_VAE
            #COST_PLOT[ni] = cost_value_vae

            # plot losses
            #plotter.make_loss_plot(COST_PLOT[:ni+1],KL_PLOT[:ni+1],params['report_interval'],fwd=False)

            if params['print_values']==True:
                print('--------------------------------------------------------------')
                print('Iteration:',i)
                print('Training Set -ELBO:',cost_value_vae)
                print('Approx KL Divergence:',KL_VAE)
                print('True KL Divergence:',old_KL)
                print('Total cost:',KL_VAE + cost_value_vae) 

        if i % params['plot_interval'] == 0 and i>0:
            for j in range(params['r']*params['r']):

                # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                XS, dt  = VICI_inverse_model.run(params, y_data_test[j].reshape([1,-1]), np.shape(x_data_test)[1],
                                                 y_normscale, 
                                                 "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
                print('Runtime to generate {} samples = {} sec'.format(params['n_samples'],dt))            

                # Make corner plot of VItamin posterior samples
                figure = corner.corner(XS, labels=params['inf_pars'],
                       quantiles=[0.16, 0.5, 0.84],
                       #range=[[0,1]]*np.shape(x_data_test)[1],
                       truths=x_data_test[j,:],
                       show_titles=True, title_kwargs={"fontsize": 12})
                plt.savefig('%s/output_%s_%d_%d.png' % (params['plot_dir'],params['run_label'],j,i))
                plt.savefig('%s/latest_%s/output_%s_%d_latest.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))            

            # Make loss plot
            plt.figure()
            xvec = params['report_interval']*np.arange(np.array(plotdata).shape[0])
            plt.semilogx(xvec,np.array(plotdata)[:,0],label='ELBO')
            plt.semilogx(xvec,np.array(plotdata)[:,1],label='KL')
            plt.semilogx(xvec,np.array(plotdata)[:,2],label='total')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.legend()
            plt.savefig('%s/cost_%s.png' % (params['plot_dir'],params['run_label']))
 
            # compare KL plot
            plt.figure()
            plt.loglog(xvec,np.array(kldata)[:,0],label='True KL')
            plt.loglog(xvec,np.array(kldata)[:,1],label='Approx KL')
            plt.loglog(xvec,-np.array(kldata)[:,1],label='Approx KL (neg)')
            plt.xlabel('iteration')
            plt.ylabel('KL')
            plt.legend()
            plt.savefig('%s/kl_comp_%s.png' % (params['plot_dir'],params['run_label']))

            # compare KL plot
            plt.figure()
            plt.semilogx(xvec,np.array(kldata)[:,0],label='True KL')
            plt.semilogx(xvec,np.array(kldata)[:,1],label='Approx KL')
            plt.xlabel('iteration')
            plt.ylabel('KL')
            plt.legend()
            plt.savefig('%s/kl_comp_%s_linear.png' % (params['plot_dir'],params['run_label']))
 
            # plot the AB histogram
            plt.figure()
            nm = 1.0/(np.exp(AB[:,0]) + np.exp(AB[:,1]))
            plt.hist(nm*np.exp(AB[:,0]),25,label='component 0')
            plt.hist(nm*np.exp(AB[:,1]),25,label='component 1')
            plt.xlabel('iteration')
            plt.ylabel('KL')
            plt.legend()
            plt.savefig('%s/latest_%s/mixweights_%s_%d_linear.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))

            # plot the AB histogram
            plt.figure()
            plt.hist(AB[:,0],25,label='component 0')
            plt.hist(AB[:,1],25,label='component 1')
            plt.xlabel('iteration')
            plt.ylabel('KL')
            plt.legend()
            plt.savefig('%s/latest_%s/mixweights_%s_%d_log.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))

        #    # Convert XS back to unnormalized version
        #    if params['do_normscale']:
        #        print(normscales)
        #        print(XS.shape)
        #        for m in range(params['ndim_x']):
        #            XS[:,m,:] = XS[:,m,:]*normscales[m]
        #
        #    # Generate final results plots
        #    plotter = plots.make_plots(params,samples,XS,pos_test)
        #
        #    # Make corner plots
        #    plotter.make_corner_plot(sampler='dynesty1')
        #
        #    # Make KL plot
        #    #plotter.gen_kl_plots(VICI_inverse_model,y_data_test,x_data,normscales)
        #
        #    Make pp plot
        #    plotter.plot_pp(VICI_inverse_model,y_data_train_l,x_data_train,0,normscales)
        #
        if i % params['save_interval'] == 0 and i > 0:
        
            # Save model 
            save_path = saver.save(session,save_dir)
                
    return            
    #return COST_PLOT, KL_PLOT, train_files 

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


def run(params, y_data_test, siz_x_data, y_normscale, load_dir):

    # USEFUL SIZES
    xsh1 = siz_x_data
    ysh1 = np.shape(y_data_test)[1]
    
    z_dimension = params['z_dimension']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']    

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-6
        
        # LOAD VICI NEURAL NETWORKS
        autoencoder = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh1, z_dimension+ysh1, n_weights_r2)
        autoencoder_ENC = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights_r1)
        autoencoder_VAE = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh1+ysh1, z_dimension, n_weights_q)
        
        # GET r(z|y)
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph")
        #y_ph_n = tf_normalise_dataset(y_ph)
        y_ph_n = y_ph
        zy_mean_a, zy_log_sig_sq_a, zy_mean_b, zy_log_sig_sq_b, ab = autoencoder_ENC._calc_z_mean_and_sigma(y_ph_n)
        
        # DRAW FROM r(z|y)
        rzy_samp = autoencoder_ENC._sample_from_gaussian_dist(tf.shape(y_ph_n)[0], z_dimension, zy_mean_a, zy_log_sig_sq_a, zy_mean_b, zy_log_sig_sq_b, ab)
        
        # GET r(x|z,y) from r(z|y) samples
        rzy_samp_y = tf.concat([rzy_samp,y_ph_n],1)
        reconstruction_xzy = autoencoder.calc_reconstruction(rzy_samp_y)
        x_mean = reconstruction_xzy[0]
        x_log_sig_sq = reconstruction_xzy[1]
        
        # GET pseudo max
        #rzy_samp_y_pm = tf.concat([zy_mean,y_ph_n],1)
        #reconstruction_xzy_pm = autoencoder.calc_reconstruction(rzy_samp_y_pm)
        #x_pmax = reconstruction_xzy_pm[0]
        
        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VICI")]
        
        # DRAW FROM q(x|y)
        qx_samp = autoencoder_VAE._sample_from_gaussian_dist(tf.shape(y_ph_n)[0], xsh1, x_mean, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq)))
        
        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(var_list_VICI)
        saver_VICI.restore(session,load_dir)
    
    # ESTIMATE TEST SET RECONSTRUCTION PER-PIXEL APPROXIMATE MARGINAL LIKELIHOOD and draw from q(x|y)
    ns = params['n_samples'] # number of samples to save per reconstruction
    #ns = np.maximum(100,n_ex_s) # number of samples to use to estimate per-pixel marginal
    
    #XM = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    #XSX = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    #XSA = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
   
    #if params['do_m1_m2_cut']: 
    #    for i in range(ns):
    #        counter = False
    #        cnt = 0
    #        while counter == False:
    #            rec_x_m = session.run(x_mean,feed_dict={y_ph:y_data_test})
    #            rec_x_mx = session.run(qx_samp,feed_dict={y_ph:y_data_test})
    #            rec_x_s = session.run(x_mean,feed_dict={y_ph:y_data_test})
    #            if (rec_x_mx[cnt,0] <= 35.0) or (rec_x_mx[cnt,0] >= 80.0) or (rec_x_mx[cnt,2] <= 35.0) or (rec_x_mx[cnt,2] >= 80.0):
    #                continue
    #            else:
    #                XSX[cnt,:,i] = rec_x_mx
    #                print('Predictions generated for test sampe %s and parameter %s ...' % (str(cnt),str(i)))
    #                cnt+=1
    #            if cnt == np.shape(y_data_test)[0]:
    #                counter = True

    #else:

    y_data_test_exp = np.tile(y_data_test,(ns,1))/y_normscale
    run_startt = time.time()
    XS = session.run(qx_samp,feed_dict={y_ph:y_data_test_exp})
    run_endt = time.time()

    #for i in range(ns):
    #    #rec_x_m = session.run(x_mean,feed_dict={y_ph:y_data_test})
    #    run_startt = time.time()
    #    rec_x_mx = session.run(qx_samp,feed_dict={y_ph:y_data_test})
    #    run_endt = time.time()
#       run_startt = time.time()
#       rec_x_s = session.run(x_mean,feed_dict={y_ph:y_data_test})
#       run_endt = time.time()

        #XM[:,:,i] = rec_x_m
    #    XSX[:,:,i] = rec_x_mx
        #XSA[:,:,i] = rec_x_s

    #pmax = None#session.run(x_pmax,feed_dict={y_ph:y_data_test})
    
    #xm = None#np.mean(XM,axis=2)
    #xsx = None#np.std(XSX,axis=2)
    #xs = None#np.std(XM,axis=2)
    #XS = XSX[:,:,0:n_ex_s]
    #XS = XSA[:,:,0:n_ex_s]

    return XS, (run_endt - run_startt)

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
