import numpy as np
import tensorflow as tf

from Neural_Networks import VICI_decoder
from Neural_Networks import VICI_encoder
from Neural_Networks import VICI_VAE_encoder
from Neural_Networks import L2_VICI_decoder
from Neural_Networks import L2_VICI_encoder
from Neural_Networks import L2_VICI_VAE_encoder
from Neural_Networks import batch_manager
from Models import CVAE

# NORMALISE DATASET FUNCTION
def tf_normalise_dataset(xp):
    
    Xs = tf.shape(xp)
    
    l2norm = tf.sqrt(tf.reduce_sum(tf.multiply(xp,xp),1))
    l2normr = tf.reshape(l2norm,[Xs[0],1])
    x_data = tf.divide(xp,l2normr)
    
    return x_data

def divide_into_patches(x_data,im_size,p_size,border):
    
    X = np.resize(x_data,(np.shape(x_data)[0],3,im_size[0],im_size[1]))
    X = np.transpose(X,(2,3,1,0))
    
    t_border = np.zeros((border,im_size[1],3,np.shape(x_data)[0]))
    l_border = np.zeros((im_size[0]+border,border,3,np.shape(x_data)[0]))
    X = np.concatenate((t_border,X),axis=0)
    X = np.concatenate((l_border,X),axis=1)
    
    np0 = np.int(np.round(im_size[0]/p_size[0]))
    np1 = np.int(np.round(im_size[1]/p_size[1]))
    
    P = np.zeros((np.shape(x_data)[0]*np0*np1,p_size[0]*p_size[1]*3))
    F = np.zeros((np.shape(x_data)[0]*np0*np1,border*p_size[1]*3+(p_size[0]+border)*border*3))
    In = np.zeros((np.shape(x_data)[0]*np0*np1,2))
    
    ni = -1
    i_f = -1.0
    for i in range(np0):
        j_f = -1.0
        i_f = i_f+1.0
        for j in range(np1):
            ni = ni+1
            j_f = j_f+1.0
            
            X_ij = X[border+i*p_size[0]:border+(i+1)*p_size[0],border+j*p_size[1]:border+(j+1)*p_size[1],:,:]
            Xt_ij = np.transpose(X_ij,(3,2,0,1))
            x_ij = np.resize(Xt_ij,(np.shape(x_data)[0],(p_size[0])*(p_size[1])*3))
            P[ni*np.shape(x_data)[0]:(ni+1)*np.shape(x_data)[0],:] = x_ij
            
            to = X[i*p_size[0]:border+i*p_size[0],border+j*p_size[1]:border+(j+1)*p_size[1],:,:]
            le = X[i*p_size[0]:border+(i+1)*p_size[0],j*p_size[1]:border+j*p_size[1],:,:]
            
            to_t = np.transpose(to,(3,2,0,1))
            t_ij = np.resize(to_t,(np.shape(x_data)[0],border*p_size[1]*3))
            le_t = np.transpose(le,(3,2,0,1))
            l_ij = np.resize(le_t,(np.shape(x_data)[0],(p_size[0]+border)*border*3))
            f = np.concatenate((t_ij,l_ij),axis=1)
            F[ni*np.shape(x_data)[0]:(ni+1)*np.shape(x_data)[0],:] = f
            
            in_ij = np.expand_dims([i_f,j_f],axis=0)
            In_ij = np.repeat(in_ij,np.shape(x_data)[0],axis=0)
            In[ni*np.shape(x_data)[0]:(ni+1)*np.shape(x_data)[0],:] = In_ij
            
    return P,F,In

def repeat_data(d,r):
    
    dn = np.zeros((r*np.shape(d)[0],np.shape(d)[1]))
    for i in range(np.shape(d)[0]):
        di = np.expand_dims(d[i,:],axis=0)
        di_r = np.repeat(di,r,axis=0)
        dn[i*r:(i+1)*r,:] = di_r
        
    return dn
        
def train(params, x_data, y_data_h, load_dir, save_dir):
    
    x_data = x_data
    y_data_train_l = y_data_h
    
    # USEFUL SIZES
    xsh = np.shape(x_data)
    ysh1 = np.shape(y_data_h)[1]
    
    p_size = 8
    border = 10
    psh1 = 3*np.square(p_size)
    bsh1 = 3*border*p_size+3*border*(p_size+border)
    
    z_dimension = params['z_dimension']
    bs = params['batch_size']
    n_weights = params['n_weights']
    
    z_dimension_p = params['z_dimensions_fw']
    n_weights_p = params['n_weights_fw']
    
    lam = 1
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-6
        
        # PLACE HOLDERS
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, psh1], name="x_ph")
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph") # batch size placeholder
        
        # LOAD VICI second layer NEURAL NETWORKS
        autoencoder = L2_VICI_decoder.VariationalAutoencoder("L2_VICI_decoder", psh1, z_dimension_p+psh1+bsh1+ysh1+2, n_weights_p)
        autoencoder_ENC = L2_VICI_encoder.VariationalAutoencoder("L2_VICI_encoder", psh1+bsh1+ysh1+2, z_dimension_p, n_weights_p)
        autoencoder_VAE = L2_VICI_VAE_encoder.VariationalAutoencoder("L2_VICI_VAE_encoder", 2*psh1+bsh1+ysh1+2, z_dimension_p, n_weights_p)
        
        # LOAD VICI NEURAL NETWORKS
        autoencoder_pre = VICI_decoder.VariationalAutoencoder("VICI_decoder", xsh[1], z_dimension+ysh1, n_weights)
        autoencoder_ENC_pre = VICI_encoder.VariationalAutoencoder("VICI_encoder", ysh1, z_dimension, n_weights)
        autoencoder_VAE_pre = VICI_VAE_encoder.VariationalAutoencoder("VICI_VAE_encoder", xsh[1]+ysh1, z_dimension, n_weights)
        
        # GET r(z|y)
#        x_ph_n = tf_normalise_dataset(x_ph)
        x_ph_n = x_ph
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, psh1+bsh1+ysh1+2], name="y_ph")
        y_ph_n = y_ph
        zy_mean,zy_log_sig_sq = autoencoder_ENC._calc_z_mean_and_sigma(y_ph_n)
        
        # DRAW FROM r(z|y)
        rzy_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension_p, zy_mean, zy_log_sig_sq)
        
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
        qzx_samp = autoencoder_VAE._sample_from_gaussian_dist(bs_ph, z_dimension_p, zx_mean, zx_log_sig_sq)
        
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
        
        ## PREVIOUS LAYER ##########################################################################################
        
        # GET r(z|y)
        y_ph_pre = tf.placeholder(dtype=tf.float32, shape=[None, ysh1], name="y_ph_pre")
        y_ph_n_pre = tf_normalise_dataset(y_ph_pre)
        zy_mean_pre,zy_log_sig_sq_pre = autoencoder_ENC_pre._calc_z_mean_and_sigma(y_ph_n_pre)
        
        # DRAW FROM r(z|y)
        rzy_samp_pre = autoencoder_VAE_pre._sample_from_gaussian_dist(bs_ph, z_dimension, zy_mean_pre, zy_log_sig_sq_pre)
        
        # GET r(x|z,y) from r(z|y) samples
        rzy_samp_y_pre = tf.concat([rzy_samp_pre,y_ph_n_pre],1)
        reconstruction_xzy_pre = autoencoder_pre.calc_reconstruction(rzy_samp_y_pre)
        x_mean_pre = reconstruction_xzy_pre[0]
        x_log_sig_sq_pre = reconstruction_xzy_pre[1]
        
        # DRAW FROM q(x|y)
        qx_samp_pre = autoencoder_ENC_pre._sample_from_gaussian_dist(bs_ph, xsh[1], x_mean_pre, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq_pre)))
        
        #########################################################################################################
        
        # THE VICI COST FUNCTION
        lam_ph = tf.placeholder(dtype=tf.float32, name="lam_ph")
        COST_VAE = KL_vae+cost_R_vae
        COST = COST_VAE
        
        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VICI")]
        var_list_VICI_L2 = [var for var in tf.trainable_variables() if var.name.startswith("L2_VICI")]
        
        # DEFINE OPTIMISER (using ADAM here)
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate']) 
        minimize = optimizer.minimize(COST,var_list = var_list_VICI_L2)
        
        # DRAW FROM q(x|y)
        qx_samp = autoencoder_ENC._sample_from_gaussian_dist(bs_ph, psh1, x_mean, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq)))
        
        # INITIALISE AND RUN SESSION        
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(var_list_VICI)
        saver_VICI.restore(session,load_dir)
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
        ynn = session.run(x_mean_pre,feed_dict={y_ph_pre:yn,bs_ph:bs})
        ynnr = repeat_data(ynn,np.int(np.round(np.square(params['image_size'][0]/p_size))))
        
        xnp, bnp, _ = divide_into_patches(x_data[next_indices, :],params['image_size'],[p_size,p_size],border)
        ynpo, _, In = divide_into_patches(ynn,params['image_size'],[p_size,p_size],border)
        
        if i<1000:
            ynpo = np.zeros((np.shape(ynpo)[0],np.shape(ynpo)[1]))
            ynnr = np.zeros((np.shape(ynnr)[0],np.shape(ynnr)[1]))
#            In = np.zeros((np.shape(In)[0],np.shape(In)[1]))
            
        ynp = np.concatenate((ynpo,bnp,ynnr,In),axis=1)
            
        session.run(minimize, feed_dict={bs_ph:np.shape(xnp)[0], x_ph:xnp,  y_ph:ynp, lam_ph:lam}) # minimising cost function
        
        if i % params['report_interval'] == 0:
                ni = ni+1
                
                ynt = y_data_train_l[0:test_n,:]
                ynnt = session.run(qx_samp_pre,feed_dict={y_ph_pre:ynt,bs_ph:test_n})
                ynntr = repeat_data(ynnt,np.int(np.round(np.square(params['image_size'][0]/p_size))))
                
                xnpt, bnpt, _ = divide_into_patches(x_data[0:test_n, :],params['image_size'],[p_size,p_size],border)
                ynpot, _, Int = divide_into_patches(ynnt,params['image_size'],[p_size,p_size],border)
                if i<1000:
                    ynpot = np.zeros((np.shape(ynpot)[0],np.shape(ynpot)[1]))
                    ynntr = np.zeros((np.shape(ynntr)[0],np.shape(ynntr)[1]))
#                    Int = np.zeros((np.shape(Int)[0],np.shape(Int)[1]))
                    
                ynpt = np.concatenate((ynpot,bnpt,ynntr,Int),axis=1)
                
                cost_value_vae, KL_VAE = session.run([COST_VAE, KL_vae], feed_dict={bs_ph:np.shape(xnpt)[0], x_ph:xnpt, y_ph:ynpt, lam_ph:lam})
                KL_PLOT[ni] = KL_VAE
                COST_PLOT[ni] = cost_value_vae
                
                if params['print_values']==True:
                    print('--------------------------------------------------------------')
                    print('Iteration:',i)
                    print('Training Set ELBO:',-cost_value_vae)
                    print('KL Divergence:',KL_VAE)
       
        if i % params['save_interval'] == 0:
             
                save_path = saver.save(session,save_dir)
                
                
    return COST_PLOT, KL_PLOT

def run(params, y_data_test, siz_x_data, load_dir_pre, load_dir):
    
    # USEFUL SIZES
    xsh1 = siz_x_data
    ysh1 = np.shape(y_data_test)[1]
    
    p_size = 8
    border = 10
    psh1 = 3*np.square(p_size)
    bsh1 = 3*border*p_size+3*border*(p_size+border)
    
#    z_dimension = params['z_dimension']
#    n_weights = params['n_weights']
    
    z_dimension_p = params['z_dimensions_fw']
    n_weights_p = params['n_weights_fw']
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-6
        
        # LOAD VICI NEURAL NETWORKS
        autoencoder = L2_VICI_decoder.VariationalAutoencoder("L2_VICI_decoder", psh1, z_dimension_p+psh1+bsh1+ysh1+2, n_weights_p)
        autoencoder_ENC = L2_VICI_encoder.VariationalAutoencoder("L2_VICI_encoder", psh1+bsh1+ysh1+2, z_dimension_p, n_weights_p)
        autoencoder_VAE = L2_VICI_VAE_encoder.VariationalAutoencoder("L2_VICI_VAE_encoder", 2*psh1+bsh1+ysh1+2, z_dimension_p, n_weights_p)
        
        # GET r(z|y)
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, psh1+bsh1+ysh1+2], name="y_ph")
        y_ph_n = y_ph
        zy_mean,zy_log_sig_sq = autoencoder_ENC._calc_z_mean_and_sigma(y_ph_n)
        
        # DRAW FROM r(z|y)
        rzy_samp = autoencoder_VAE._sample_from_gaussian_dist(tf.shape(y_ph_n)[0], z_dimension_p, zy_mean, zy_log_sig_sq)
        
        # GET r(x|z,y) from r(z|y) samples
        rzy_samp_y = tf.concat([rzy_samp,y_ph_n],1)
        reconstruction_xzy = autoencoder.calc_reconstruction(rzy_samp_y)
        x_mean = reconstruction_xzy[0]
        x_log_sig_sq = reconstruction_xzy[1]
        
        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("L2_VICI")]
        
        # DRAW FROM q(x|y)
        qx_samp = autoencoder_ENC._sample_from_gaussian_dist(tf.shape(y_ph_n)[0], psh1, x_mean, SMALL_CONSTANT + tf.log(tf.exp(x_log_sig_sq)))
        
        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(var_list_VICI)
        saver_VICI.restore(session,load_dir)
    
    # ESTIMATE TEST SET RECONSTRUCTION PER-PIXEL APPROXIMATE MARGINAL LIKELIHOOD and draw from q(x|y)
    ns = 10 # number of samples to use to estimate per-pixel marginal
    n_ex_s = params['n_samples'] # number of samples to save per reconstruction
    
    XM = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    XSX = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    XSA = np.zeros((np.shape(y_data_test)[0],xsh1,ns))
    
    for i in range(ns):
        
        _, _, XS_c = CVAE.run(params, y_data_test, siz_x_data, load_dir_pre)
        ynn = np.squeeze(XS_c[:,:,0])
        
        Y = np.resize(ynn,(np.shape(y_data_test)[0],3,params['image_size'][0],params['image_size'][1]))
        Y = np.transpose(Y,(2,3,1,0))
        
        n1 = np.int(np.round(params['image_size'][0]/p_size))
        n2 = np.int(np.round(params['image_size'][1]/p_size))
        
        X = np.zeros((np.shape(y_data_test)[0],3,params['image_size'][0],params['image_size'][1]))
        X = np.transpose(X,(2,3,1,0))
        t_border = np.zeros((border,params['image_size'][1],3,np.shape(y_data_test)[0]))
        l_border = np.zeros((params['image_size'][0]+border,border,3,np.shape(y_data_test)[0]))
        X = np.concatenate((t_border,X),axis=0)
        X = np.concatenate((l_border,X),axis=1)
        
        Xs = np.zeros((np.shape(y_data_test)[0],3,params['image_size'][0],params['image_size'][1]))
        Xs = np.transpose(Xs,(2,3,1,0))
        Xs = np.concatenate((t_border,Xs),axis=0)
        Xs = np.concatenate((l_border,Xs),axis=1)
        
        Y = np.concatenate((t_border,Y),axis=0)
        Y = np.concatenate((l_border,Y),axis=1)
        
        j1_f = -1.0
        for j1 in range(n1):
            j2_f = -1.0
            j1_f = j1_f+1.0
            for j2 in range(n2):
                j2_f = j2_f+1.0
                
                Y_ij = Y[border+j1*p_size:border+(j1+1)*p_size,border+j2*p_size:border+(j2+1)*p_size,:,:]
                Yt_ij = np.transpose(Y_ij,(3,2,0,1))
                y1_ij = np.resize(Yt_ij,(np.shape(y_data_test)[0],(p_size)*(p_size)*3))
                
                to = X[j1*p_size:border+j1*p_size,border+j2*p_size:border+(j2+1)*p_size,:,:]
                le = X[j1*p_size:border+(j1+1)*p_size,j2*p_size:border+j2*p_size,:,:]
                
                to_t = np.transpose(to,(3,2,0,1))
                t_ij = np.resize(to_t,(np.shape(y_data_test)[0],border*p_size*3))
                le_t = np.transpose(le,(3,2,0,1))
                l_ij = np.resize(le_t,(np.shape(y_data_test)[0],(p_size+border)*border*3))
                y2_ij = np.concatenate((t_ij,l_ij),axis=1)
                
#                toy = Y[j1*p_size:border+j1*p_size,border+j2*p_size:border+(j2+1)*p_size,:,:]
#                ley = Y[j1*p_size:border+(j1+1)*p_size,j2*p_size:border+j2*p_size,:,:]
#                
#                toy_t = np.transpose(toy,(3,2,0,1))
#                ty_ij = np.resize(toy_t,(np.shape(y_data_test)[0],border*p_size*3))
#                ley_t = np.transpose(ley,(3,2,0,1))
#                ly_ij = np.resize(ley_t,(np.shape(y_data_test)[0],(p_size+border)*border*3))
#                y3_ij = np.concatenate((ty_ij,ly_ij),axis=1)
                y3_ij = ynn
                
                in_ij = np.expand_dims([j1_f,j2_f],axis=0)
                In_ij = np.repeat(in_ij,np.shape(y_data_test)[0],axis=0)
                
#                y1_ij = np.zeros((np.shape(y1_ij)[0],np.shape(y1_ij)[1]))
#                y3_ij = np.zeros((np.shape(y3_ij)[0],np.shape(y3_ij)[1]))
                
                y_ij = np.concatenate((y1_ij,y2_ij,y3_ij,In_ij),axis=1)
                
                rec_x_m_ij = session.run(x_mean,feed_dict={y_ph:y_ij})
                rec_x_mx_ij = session.run(qx_samp,feed_dict={y_ph:y_ij})
                
                X_mij = np.resize(rec_x_m_ij,(np.shape(y_data_test)[0],3,p_size,p_size))
                X_mij = np.transpose(X_mij,(2,3,1,0))
                X[border+j1*p_size:border+(j1+1)*p_size,border+j2*p_size:border+(j2+1)*p_size,:,:] = X_mij
                
                Xs_mij = np.resize(rec_x_mx_ij,(np.shape(y_data_test)[0],3,p_size,p_size))
                Xs_mij = np.transpose(Xs_mij,(2,3,1,0))
                Xs[border+j1*p_size:border+(j1+1)*p_size,border+j2*p_size:border+(j2+1)*p_size,:,:] = Xs_mij
        
        X = X[border:np.shape(X)[0],border:np.shape(X)[1],:,:]
        Xt = np.transpose(X,(3,2,0,1))
        rec_x_m = np.resize(Xt,(np.shape(y_data_test)[0],3*params['image_size'][0]*params['image_size'][1]))
        
        Xs = Xs[border:np.shape(Xs)[0],border:np.shape(Xs)[1],:,:]
        Xst = np.transpose(Xs,(3,2,0,1))
        rec_x_mx = np.resize(Xst,(np.shape(y_data_test)[0],3*params['image_size'][0]*params['image_size'][1]))
        
        XM[:,:,i] = rec_x_m
        XSX[:,:,i] = rec_x_mx
        XSA[:,:,i] = rec_x_m
    
    xm = np.mean(XM,axis=2)
    xsx = np.std(XSX,axis=2)
    xs = np.std(XM,axis=2)
    XS = XSA[:,:,0:n_ex_s]
    
                
    return xm, xsx, XS

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
        saver_VICI.restore(session,load_dir)
                
    ynt = y_data_h
    cost_value_vae, KL_VAE = session.run([COST_VAE, KL_vae], feed_dict={bs_ph:xsh[0], x_ph:x_data, y_ph:ynt, lam_ph:lam})
    ELBO = -cost_value_vae
    KL_DIV = KL_VAE
                
    return ELBO, KL_DIV