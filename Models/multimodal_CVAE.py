import numpy as np
import tensorflow as tf

from Neural_Networks import VICI_decoder
from Neural_Networks import VICI_encoder
from Neural_Networks import VICI_VAE_encoder
from Neural_Networks import batch_manager

# NORMALISE DATASET FUNCTION
def tf_normalise_dataset(xp):
    
    Xs = tf.shape(xp)
    
    l2norm = tf.sqrt(tf.reduce_sum(tf.multiply(xp,xp),1))
    l2normr = tf.reshape(l2norm,[Xs[0],1])
    x_data = tf.divide(xp,l2normr)
    
    return x_data

# MULTIMODAL UPDATE: NORMALISE DATASET TO THE SUM FUNCTION
def tf_normalise_sum_dataset(xp):
    
    Xs = tf.shape(xp)
    
    norm = tf.reduce_sum(xp,1)
    normr = tf.reshape(norm,[Xs[0],1])
    x_data = tf.divide(xp,normr)
    
    return x_data

# MULTIMODAL UPDATE: repeat a tensor
def tf_repeat(tensor, repeats):
   
    with tf.variable_scope("repeat"):
#        expanded_tensor = tf.expand_dims(tensor, -1)
#        multiples = [1] + repeats
#        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
#        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
        repeated_tesnor = tf.tile(tensor,repeats)
    return repeated_tesnor

def select_with_prob(A,B,P):
    
    P0 = tf.expand_dims(P[0,:],1)
    P0 = tf_repeat(P0, [1,tf.shape(A,1),1])
    P1 = tf.expand_dims(P[1,:],1)
    P1 = tf_repeat(P1, [1,tf.shape(A,1),1])
    
    S0 = tf.sigmoid((tf.random_uniform(tf.shape(A),0.0,1.0)-(1-P0))*200)
    S1 = tf.sigmoid((tf.random_uniform(tf.shape(A),0.0,1.0)-(1-P1))*200)
    
    C = tf.multiply(S0,A) + tf.multiply(S1,B)
    
    return C, S0, S1

def train(params, x_data, y_data_h, save_dir):
    
    x_data = x_data
    y_data_train_l = y_data_h
    
    # USEFUL SIZES
    xsh = np.shape(x_data)
    ysh1 = np.shape(y_data_h)[1]
    
    z_dimension = params['z_dimension']
    bs = params['batch_size']
    n_weights = params['n_weights']
    lam = 10
    
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
        # MULTIMODAL UPDATE: making several encoders (e.g. 2)
        autoencoder_ENC = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights)
        autoencoder_ENC_1 = VICI_encoder.VariationalAutoencoder("VICI_encoder_1", ysh1, z_dimension, n_weights)
        # MULTIMODAL UPDATE: making model to infer the probabilities in the mixture
        infer_pm = VICI_encoder.VariationalAutoencoder("VICI_infer_pm", ysh1, 2, n_weights) # the "2" is because we are trying with 2 Gaussians. Change accordingly
        infer_pm_q = VICI_encoder.VariationalAutoencoder("VICI_infer_pm_q", xsh[1], 2, n_weights) # the "2" is because we are trying with 2 Gaussians. Change accordingly
        
        autoencoder_VAE = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh[1]+ysh1, z_dimension, n_weights)
        
        # GET r(z|y)
        x_ph_n = tf_normalise_dataset(x_ph)
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph")
        y_ph_n = tf_normalise_dataset(y_ph)
        # MULTIMODAL UPDATE: inferring moments of the different distributions
        zy_mean,zy_log_sig_sq = autoencoder_ENC._calc_z_mean_and_sigma(y_ph_n)
        zy_mean_1,zy_log_sig_sq_1 = autoencoder_ENC_1._calc_z_mean_and_sigma(y_ph_n)
        # MULTIMODAL UPDATE: inferring weights of the different distributions in the mixture
        wm,_ = infer_pm._calc_z_mean_and_sigma(y_ph_n)
        wm = tf_normalise_sum_dataset(wm)
        wm_q,_ = infer_pm_q._calc_z_mean_and_sigma(x_ph_n)
        wm_q = tf_normalise_sum_dataset(wm_q)
        
        # DRAW FROM r(z|y)
        rzy_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension, zy_mean, zy_log_sig_sq)
        # MULTIMODAL UPDATE: DRAW FROM r_1(z|y)
        rzy_samp_1 = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension, zy_mean_1, zy_log_sig_sq_1)
        
        # GET r(x|z,y) from r(z|y) samples
        rzy_samp_y = tf.concat([rzy_samp,y_ph_n],1)
        reconstruction_xzy = autoencoder.calc_reconstruction(rzy_samp_y)
        x_mean = reconstruction_xzy[0]
        x_log_sig_sq = reconstruction_xzy[1]
        # MULTIMODAL UPDATE: same for second distribution
        rzy_samp_y_1 = tf.concat([rzy_samp_1,y_ph_n],1)
        reconstruction_xzy_1 = autoencoder.calc_reconstruction(rzy_samp_y_1)
        x_mean_1 = reconstruction_xzy_1[0]
        x_log_sig_sq_1 = reconstruction_xzy_1[1]
        
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
        KL_vae_0 = tf.reduce_mean(tf.multiply(wm_q[:,0],cost_VAE_b))
        # MULTIMODAL UPDATE: KL divergence with second gaussian in the mixture
        v_mean = zy_mean_1 #2
        aux_mean = zx_mean #1
        v_log_sig_sq = tf.log(tf.exp(zy_log_sig_sq_1)+SMALL_CONSTANT) #2
        aux_log_sig_sq = tf.log(tf.exp(zx_log_sig_sq)+SMALL_CONSTANT) #1
        v_log_sig = tf.log(tf.sqrt(tf.exp(v_log_sig_sq))) #2
        aux_log_sig = tf.log(tf.sqrt(tf.exp(aux_log_sig_sq))) #1
        cost_VAE_a = v_log_sig-aux_log_sig+tf.divide(tf.exp(aux_log_sig_sq)+tf.square(aux_mean-v_mean),2*tf.exp(v_log_sig_sq))-0.5
        cost_VAE_b = tf.reduce_sum(cost_VAE_a,1)
        KL_vae_1 = tf.reduce_mean(tf.multiply(wm_q[:,1],cost_VAE_b))
        # MULTIMODAL UPDATE: KL divergence between weights
        latent_loss_wm = tf.reduce_sum(wm_q*tf.log(wm_q)-wm_q*tf.log(wm),1)
        KL_wm = tf.reduce_mean(latent_loss_wm)
        # MULTIMODAL UPDATE: total KL divergence
        KL_vae = KL_vae_0 + KL_vae_1 + KL_wm
        
        # THE VICI COST FUNCTION
        lam_ph = tf.placeholder(dtype=tf.float32, name="lam_ph")
        COST_VAE = KL_vae+cost_R_vae
        COST = COST_VAE
        
        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VICI")]
        
        # DEFINE OPTIMISER (using ADAM here)
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate']) 
        minimize = optimizer.minimize(COST,var_list = var_list_VICI)
        
        # DRAW FROM q(x|y)
        qx_samp = autoencoder_ENC._sample_from_gaussian_dist(bs_ph, xsh[1], x_mean, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq)))
        
        # INITIALISE AND RUN SESSION
#        init = tf.variables_initializer(var_list_VICI)
        init = tf.initialize_all_variables()
        session.run(init)
        saver = tf.train.Saver()
    
    KL_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1)) # vector to store test OELBO values
    COST_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])+1)) # vector to store test VAE ELBO values
    
    print('Training CVAE Inference Model...')    
    # START OPTIMISATION OF OELBO
    indices_generator = batch_manager.SequentialIndexer(params['batch_size'], xsh[0])
    ni = -1
    test_n = 100
    for i in range(params['num_iterations']):
        
        next_indices = indices_generator.next_indices()
        
        yn = y_data_train_l[next_indices, :]
        session.run(minimize, feed_dict={bs_ph:bs, x_ph:x_data[next_indices, :],  y_ph:yn, lam_ph:lam}) # minimising cost function
        
        if i % params['report_interval'] == 0:
                ni = ni+1
                
                ynt = y_data_train_l[0:test_n,:]
                cost_value_vae, KL_VAE = session.run([COST_VAE, KL_vae], feed_dict={bs_ph:test_n, x_ph:x_data[0:test_n,:], y_ph:ynt, lam_ph:lam})
                KL_PLOT[ni] = KL_VAE
                COST_PLOT[ni] = cost_value_vae
                
                if params['print_values']==True:
                    print('--------------------------------------------------------------')
                    print('Iteration:',i)
                    print('Test Set ELBO:',-cost_value_vae)
                    print('KL Divergence:',KL_VAE)
       
        if i % params['save_interval'] == 0:
             
                save_path = saver.save(session,save_dir)
                
                
    return COST_PLOT, KL_PLOT

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
        autoencoder = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh[1], z_dimension+ysh1, n_weights)
        # MULTIMODAL UPDATE: making several encoders (e.g. 2)
        autoencoder_ENC = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights)
        autoencoder_ENC_1 = VICI_encoder.VariationalAutoencoder("VICI_encoder_1", ysh1, z_dimension, n_weights)
        # MULTIMODAL UPDATE: making model to infer the probabilities in the mixture
        infer_pm = VICI_encoder.VariationalAutoencoder("VICI_infer_pm", ysh1, 2, n_weights) # the "2" is because we are trying with 2 Gaussians. Change accordingly
        infer_pm_q = VICI_encoder.VariationalAutoencoder("VICI_infer_pm_q", xsh[1], 2, n_weights) # the "2" is because we are trying with 2 Gaussians. Change accordingly
        
        autoencoder_VAE = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh[1]+ysh1, z_dimension, n_weights)
        
        # GET r(z|y)
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph")
        y_ph_n = tf_normalise_dataset(y_ph)
        # MULTIMODAL UPDATE: inferring moments of the different distributions
        zy_mean,zy_log_sig_sq = autoencoder_ENC._calc_z_mean_and_sigma(y_ph_n)
        zy_mean_1,zy_log_sig_sq_1 = autoencoder_ENC_1._calc_z_mean_and_sigma(y_ph_n)
        # MULTIMODAL UPDATE: inferring weights of the different distributions in the mixture
        wm,_ = infer_pm._calc_z_mean_and_sigma(y_ph_n)
        wm = tf_normalise_sum_dataset(wm)
        
        # DRAW FROM r(z|y)
        _,S0,S1 = select_with_prob(zy_mean_1,zy_mean,wm)
        rzy_samp = autoencoder_VAE._sample_from_gaussian_dist(tf.shape(y_ph_n)[0], z_dimension, tf.multiply(S0,zy_mean)+tf.multiply(S1,zy_mean_1), tf.multiply(S0,zy_log_sig_sq)+tf.multiply(S1,zy_log_sig_sq_1))
        
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
    ns = 100 # number of samples to use to estimate per-pixel marginal
    n_ex_s = params['n_samples'] # number of samples to save per reconstruction
    
    XM = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    XSX = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    XSA = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    
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
    XS = XSA[:,:,0:n_ex_s]
    
                
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