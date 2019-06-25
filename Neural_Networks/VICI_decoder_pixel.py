import collections

import tensorflow as tf
import numpy as np
#import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

class VariationalAutoencoder(object):

    def __init__(self, name, n_input, n_hidden, n_weights, siz_image, siz_filt, middle="gaussian"):
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_weights = n_weights
        self.siz_image = siz_image
        self.siz_filt = siz_filt
        self.name = name
        self.middle = middle
        self.bias_start = 0.0

        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinearity = tf.nn.relu
     
    def tf_repeat(self, tensor, repeats):
        """
        Args:
    
        input: A Tensor. 1-D or higher.
        repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
    
        Returns:
        
        A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
        """
        with tf.variable_scope("repeat"):
            expanded_tensor = tf.expand_dims(tensor, -1)
            multiples = [1] + repeats
            tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
            repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
        return repeated_tesnor    
    
    def conv_2D(self,x,d,m):
        
        msh = tf.shape(m)
        xsh0 = tf.shape(x)[0]
        d1 = d[0]
        d2 = d[1]
#        X = tf.reshape(x,[xsh0,d1,d2,3,1])
        X = tf.reshape(x,[xsh0,3,d1,d2,1])
#        x_o = x_sample[i+offset].reshape(3, 32, 32)
        X = tf.transpose(X, perm=(0,2,3,1,4))
        Xr = X[:,:,:,0,:]
        Xg = X[:,:,:,1,:]
        Xb = X[:,:,:,2,:]
        M = tf.reshape(m,[msh[0],msh[1],1,1])
        Yr = tf.nn.conv2d(Xr, M,padding='SAME',strides=[1, 1, 1, 1])
        Yg = tf.nn.conv2d(Xg, M,padding='SAME',strides=[1, 1, 1, 1])
        Yb = tf.nn.conv2d(Xb, M,padding='SAME',strides=[1, 1, 1, 1])
#        Y = tf.concat([Yr,Yg,Yb],2)
        Y = tf.concat([Yr,Yg,Yb],3)
        Y = tf.transpose(Y, perm=(0,3,1,2))
        y = tf.reshape(Y,[xsh0,d[0]*d[1]*3])
        
        return y
    
    def conv2D_iteration(self,x,d,m,inde):
        
        inde = inde[0]
        msh = tf.shape(m)
        msh2 = tf.cast(tf.floor(tf.divide(tf.cast(msh,tf.int8),2)), tf.int32)
#        msh2 = tf.floor(tf.divide(msh,2))
        indf = inde+msh2
        ind_st = indf-msh2
        ind_end = indf+msh2
        
        xsh0 = tf.shape(x)[0]
        d1 = d[0]
        d2 = d[1]
#        X = tf.reshape(x,[xsh0,d1,d2,3,1])
        X = tf.reshape(x,[xsh0,3,d1,d2,1])
#        x_o = x_sample[i+offset].reshape(3, 32, 32)
        X = tf.transpose(X, perm=(0,2,3,1,4)) # [batch_size,d1,d2,RGB,1]
        X = tf.pad(X,[[0,0],[msh2[0],msh2[0]],[msh2[1],msh2[1]],[0,0],[0,0]])
        Xr = X[:,:,:,0,:]
        Xg = X[:,:,:,1,:]
        Xb = X[:,:,:,2,:]
        
        l_end0 = tf.shape(X)[1]-ind_end[0]-1
        l_end1 = tf.shape(X)[2]-ind_end[1]-1
        mb1 = tf.zeros([tf.shape(X)[1],ind_st[1]])
        mb2a = tf.zeros([ind_st[0],msh[1]])
        mb2b = m
        mb2c = tf.zeros([l_end0,msh[1]])
        mb2 = tf.concat([mb2a,mb2b,mb2c],0)
        mb3 = tf.zeros([tf.shape(X)[1],l_end1])
        mb = tf.concat([mb1,mb2,mb3],1)
        
        Msh = tf.shape(mb)
        M = tf.reshape(mb,[Msh[0],Msh[1],1,1])
        M = self.tf_repeat(M,[1,1,tf.shape(X)[0],1])
        M = tf.transpose(M, perm=(2,0,1,3))
        Yr = tf.multiply(Xr, M)
        Yr = Yr[:,msh2[0]+1:msh2[0]+d1+1,msh2[1]+1:msh2[1]+d2+1,:]
        Yg = tf.multiply(Xg, M)
        Yg = Yg[:,msh2[0]+1:msh2[0]+d1+1,msh2[1]+1:msh2[1]+d2+1,:]
        Yb = tf.multiply(Xb, M)
        Yb = Yb[:,msh2[0]+1:msh2[0]+d1+1,msh2[1]+1:msh2[1]+d2+1,:]
#        Y = tf.concat([Yr,Yg,Yb],2)
        Y = tf.concat([Yr,Yg,Yb],3)
        Y = tf.transpose(Y, perm=(0,3,1,2))
        y = tf.reshape(Y,[xsh0,d[0]*d[1]*3])
        
        return y
        
    def calc_reconstruction(self, z):
        with tf.name_scope("VICI_decoder"):
            if self.middle == "bernoulli":
                hidden1_pre = tf.add(tf.matmul(z, self.weights['ELBO_decoder']['W1_to_hidden']), self.weights['ELBO_decoder']['b1_to_hidden'])
                hidden1_post = self.nonlinearity(hidden1_pre)
                
                hidden3_pre = tf.add(tf.matmul(hidden1_post, self.weights['ELBO_decoder']['W1c_htoh']), self.weights['ELBO_decoder']['b1c_htoh'])
                hidden3_post = self.nonlinearity(hidden3_pre)

                hidden2_pre = tf.add(tf.matmul(hidden3_post, self.weights['ELBO_decoder']['W1b_htoh']), self.weights['ELBO_decoder']['b1b_htoh'])
                hidden2_post = self.nonlinearity(hidden2_pre)

                y_pre = tf.add(tf.matmul(hidden2_post, self.weights['ELBO_decoder']['W2_to_y_pre']), self.weights['ELBO_decoder']['b2_to_y_pre'])
                y = tf.sigmoid(y_pre)
                return y
            elif self.middle == "gaussian":
                hidden1_pre = tf.add(tf.matmul(z,self.weights['VICI_decoder']['W3_to_hiddenG']), self.weights['VICI_decoder']['b3_to_hiddenG'])
                hidden1_post = self.nonlinearity(hidden1_pre)
#                hidden1_post = tf.nn.batch_normalization(hidden1_post,tf.Variable(tf.zeros([400], dtype=tf.float32)),tf.Variable(tf.ones([400], dtype=tf.float32)),None,None,0.000001,name="d_b_norm_1")

#                hidden1_pre_s = tf.add(tf.matmul(z,self.weights['decoder']['W3_to_hiddenGS']), self.weights['decoder']['b3_to_hiddenGS'])
#                hidden1_post_s = self.nonlinearity(hidden1_pre_s)

#                hidden1c_pre = tf.add(tf.matmul(hidden1_post, self.weights['decoder']['W3c_to_hiddenG']), self.weights['decoder']['b3c_to_hiddenG'])
#                hidden1c_post = self.nonlinearity(hidden1c_pre)
##                
#                hidden1d_pre = tf.add(tf.matmul(hidden1c_post, self.weights['decoder']['W3d_to_hiddenG']), self.weights['decoder']['b3d_to_hiddenG'])
#                hidden1d_post = self.nonlinearity(hidden1d_pre)
                
#                hidden1e_pre = tf.add(tf.matmul(hidden1d_post, self.weights['decoder']['W3e_to_hiddenG']), self.weights['decoder']['b3e_to_hiddenG'])
#                hidden1e_post = self.nonlinearity(hidden1e_pre)
                
#                hidden1b_pre = tf.add(tf.matmul(hidden1_post, self.weights['decoder']['W3b_to_hiddenG']), self.weights['decoder']['b3b_to_hiddenG'])
#                hidden1b_post = self.nonlinearity(hidden1b_pre)
#                hidden1b_post = hidden1_post

                y = tf.add(tf.matmul(hidden1_post, self.weights['VICI_decoder']['W4_to_y']), self.weights['VICI_decoder']['b4_to_y'])
                y = tf.sigmoid(y)  # see paper
#                mu = tf.sigmoid(mu+0.5)-0.5  # see paper
                return y
            else:
                RuntimeError(self.middle + " is not yet constructed for reconstruction")
                
    def pixel_probability(self, y, x):
        
        mc = self.weights['VICI_decoder']['conv_to_c']
        sf = tf.cast(self.siz_filt, tf.int32)
        hsf = tf.cast(tf.floor(tf.divide(tf.cast(self.siz_filt,tf.int8),2)), tf.int32)
        
        msk1 = tf.ones([hsf[0],sf[1]])
        msk2a = tf.ones([1,hsf[1]])
        msk2b = tf.zeros([1,hsf[1]+1])
        msk2 = tf.concat([msk2a,msk2b],1)
        msk3 = tf.zeros([hsf[0],sf[1]])
        msk = tf.concat([msk1,msk2,msk3],0)
        m = tf.multiply(mc,msk)
        
        W_c_to_mu = self.tf_repeat(self.weights['VICI_decoder']['c_to_mu'],[tf.shape(x)[0],1])
        W_y_to_mu = self.tf_repeat(self.weights['VICI_decoder']['c_to_mu'],[tf.shape(x)[0],1])
        W_c_to_sigma = self.tf_repeat(self.weights['VICI_decoder']['c_to_sigma'],[tf.shape(x)[0],1])
        W_y_to_sigma = self.tf_repeat(self.weights['VICI_decoder']['c_to_sigma'],[tf.shape(x)[0],1])
        
        c = self.conv_2D(x,self.siz_image,m)
        
        mu_pre = tf.add(tf.multiply(c, W_c_to_mu), tf.multiply(y, W_y_to_mu))
        mu = tf.add(mu_pre,self.weights['VICI_decoder']['b_to_mu'])
        mu = tf.sigmoid(mu)
        
        sigma_pre = tf.add(tf.multiply(c, W_c_to_sigma), tf.multiply(y, W_y_to_sigma))
        sigma = tf.add(sigma_pre,self.weights['VICI_decoder']['b_to_sigma'])
        
        return mu, sigma
    
    def run_pixel_probability(self, y, x, inde):
        
        mc = self.weights['VICI_decoder']['conv_to_c']
        sf = tf.cast(self.siz_filt, tf.int32)
        hsf = tf.cast(tf.floor(tf.divide(tf.cast(self.siz_filt,tf.int8),2)), tf.int32)
        
        msk1 = tf.ones([hsf[0],sf[1]])
        msk2a = tf.ones([1,hsf[1]])
        msk2b = tf.zeros([1,hsf[1]+1])
        msk2 = tf.concat([msk2a,msk2b],1)
        msk3 = tf.zeros([hsf[0],sf[1]])
        msk = tf.concat([msk1,msk2,msk3],0)
        m = tf.multiply(mc,msk)
        
        W_c_to_mu = self.tf_repeat(self.weights['VICI_decoder']['c_to_mu'],[tf.shape(x)[0],1])
        W_y_to_mu = self.tf_repeat(self.weights['VICI_decoder']['c_to_mu'],[tf.shape(x)[0],1])
        W_c_to_sigma = self.tf_repeat(self.weights['VICI_decoder']['c_to_sigma'],[tf.shape(x)[0],1])
        W_y_to_sigma = self.tf_repeat(self.weights['VICI_decoder']['c_to_sigma'],[tf.shape(x)[0],1])
        
        c = self.conv2D_iteration(x,self.siz_image,m,inde)
        
        mu_pre = tf.add(tf.multiply(c, W_c_to_mu), tf.multiply(y, W_y_to_mu))
        mu = tf.add(mu_pre,self.weights['VICI_decoder']['b_to_mu'])
        mu = tf.sigmoid(mu)
        
        sigma_pre = tf.add(tf.multiply(c, W_c_to_sigma), tf.multiply(y, W_y_to_sigma))
        sigma = tf.add(sigma_pre,self.weights['VICI_decoder']['b_to_sigma'])
        
        return mu, sigma

    def _create_weights(self):
        all_weights = collections.OrderedDict()

        # Decoder
        with tf.variable_scope("VICI_DEC"):
            all_weights['VICI_decoder'] = collections.OrderedDict()
            if self.middle == "gaussian":
                hidden_number_decoder = self.n_weights
                sf = self.siz_filt
                all_weights['VICI_decoder']['W3_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(self.n_hidden, hidden_number_decoder), dtype=tf.float32)
                all_weights['VICI_decoder']['b3_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
                
    #            all_weights['decoder']['W3_to_hiddenGS'] = tf.Variable(vae_utils.xavier_init(self.n_hidden, hidden_number_decoder), dtype=tf.float32)
    #            all_weights['decoder']['b3_to_hiddenGS'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
                
    #            all_weights['decoder']['W3b_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
    #            all_weights['decoder']['b3b_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
    #            
    #            all_weights['decoder']['W3c_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
    #            all_weights['decoder']['b3c_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
    #####            
    #            all_weights['decoder']['W3d_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
    #            all_weights['decoder']['b3d_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
    #            
    #            all_weights['decoder']['W3e_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
    #            all_weights['decoder']['b3e_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
                
                all_weights['VICI_decoder']['conv_to_c'] = tf.Variable(vae_utils.xavier_init(sf[0], sf[1]), dtype=tf.float32)
                
                all_weights['VICI_decoder']['c_to_mu'] = tf.Variable(vae_utils.xavier_init(1, self.n_input), dtype=tf.float32)
                all_weights['VICI_decoder']['c_to_sigma'] = tf.Variable(vae_utils.xavier_init(1, self.n_input), dtype=tf.float32)
                all_weights['VICI_decoder']['y_to_mu'] = tf.Variable(vae_utils.xavier_init(1, self.n_input), dtype=tf.float32)
                all_weights['VICI_decoder']['y_to_sigma'] = tf.Variable(vae_utils.xavier_init(1, self.n_input), dtype=tf.float32)
                all_weights['VICI_decoder']['b_to_mu'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
#                all_weights['VICI_decoder']['conv_to_sigma'] = tf.Variable(vae_utils.xavier_init(sf[0], sf[1]), dtype=tf.float32)
                all_weights['VICI_decoder']['b_to_sigma'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
    
                all_weights['VICI_decoder']['W4_to_y'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input), dtype=tf.float32)
                all_weights['VICI_decoder']['b4_to_y'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
                all_weights['VICI_decoder']['W5_to_log_sigmaG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input), dtype=tf.float32)
#                all_weights['VICI_decoder']['b5_to_log_sigmaG'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
            elif self.middle == "bernoulli":
                hidden_number_decoder = 200
                all_weights['decoder']['W1_to_hidden'] = tf.Variable(vae_utils.xavier_init(self.n_hidden, hidden_number_decoder))
                tf.summary.histogram("W1_to_hidden", all_weights['decoder']['W1_to_hidden'])
    
                all_weights['decoder']['W1b_htoh'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder))
                tf.summary.histogram("W1b_htoh", all_weights['decoder']['W1_to_hidden'])
                
                all_weights['decoder']['W1c_htoh'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder))
                tf.summary.histogram("W1c_htoh", all_weights['decoder']['W1_to_hidden'])
    
                all_weights['decoder']['W2_to_y_pre'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input))
                tf.summary.histogram("W2_to_y_pre", all_weights['decoder']['W1_to_hidden'])
    
                all_weights['decoder']['b1_to_hidden'] = tf.Variable(tf.ones([hidden_number_decoder], dtype=tf.float32) * self.bias_start)
                tf.summary.histogram("b1_to_hidden", all_weights['decoder']['b1_to_hidden'])
    
                all_weights['decoder']['b1b_htoh'] = tf.Variable(tf.ones([hidden_number_decoder], dtype=tf.float32) * self.bias_start)
                tf.summary.histogram("b1b_htoh", all_weights['decoder']['b1b_htoh'])
                
                all_weights['decoder']['b1c_htoh'] = tf.Variable(tf.ones([hidden_number_decoder], dtype=tf.float32) * self.bias_start)
                tf.summary.histogram("b1c_htoh", all_weights['decoder']['b1c_htoh'])
    
                all_weights['decoder']['b2_to_y_pre'] = tf.Variable(tf.ones([self.n_input], dtype=tf.float32) * self.bias_start)
                tf.summary.histogram("b2_to_y_pre", all_weights['decoder']['b2_to_y_pre'])
    
            else:
                raise RuntimeError
            
            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights