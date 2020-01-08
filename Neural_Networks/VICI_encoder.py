import collections

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6
tfd = tfp.distributions

class VariationalAutoencoder(object):

    def __init__(self, name, n_input, n_hidden, n_weights, middle="gaussian"):
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_weights = n_weights
        self.name = name
        self.middle = middle
        self.bias_start = 0.0
        self.drate = 0.0

        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinearity = tf.nn.relu
        #self.nonlinearity = tf.nn.leaky_relu


    def _calc_z_mean_and_sigma(self,x):
        with tf.name_scope("VICI_encoder"):
            hidden1_pre = tf.add(tf.matmul(x, self.weights['VICI_encoder']['W3_to_hidden']), self.weights['VICI_encoder']['b3_to_hidden'])
            hidden1_post = self.nonlinearity(hidden1_pre)
            hidden1_dropout = tf.layers.dropout(hidden1_post,rate=self.drate)
#            hidden1_post = tf.nn.batch_normalization(hidden1_post,tf.Variable(tf.zeros([400], dtype=tf.float32)),tf.Variable(tf.ones([400], dtype=tf.float32)),None,None,0.000001,name="e_b_norm_1")

            hidden2_pre = tf.add(tf.matmul(hidden1_dropout, self.weights['VICI_encoder']['W3_hth']), self.weights['VICI_encoder']['b3_hth'])
            hidden2_post = self.nonlinearity(hidden2_pre)
            hidden2_dropout = tf.layers.dropout(hidden2_post,rate=self.drate)

            hidden3_pre = tf.add(tf.matmul(hidden2_dropout, self.weights['VICI_encoder']['W3b_hth']), self.weights['VICI_encoder']['b3b_hth'])
            hidden3_post = self.nonlinearity(hidden3_pre)
            hidden3_dropout = tf.layers.dropout(hidden3_post,rate=self.drate)
##            
#            hidden4_pre = tf.add(tf.matmul(hidden3_post, self.weights['IVA_encoder']['W3c_hth']), self.weights['IVA_encoder']['b3c_hth'])
#            hidden4_post = self.nonlinearity(hidden4_pre)
#            
#            hidden5_pre = tf.add(tf.matmul(hidden4_post, self.weights['encoder']['W3d_hth']), self.weights['encoder']['b3d_hth'])
#            hidden5_post = self.nonlinearity(hidden5_pre)
            
#            hidden2_post = hidden1_post

            z_mean_a = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_encoder']['W4_to_mu_a']), self.weights['VICI_encoder']['b4_to_mu_a'])
            z_mean_b = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_encoder']['W4_to_mu_b']), self.weights['VICI_encoder']['b4_to_mu_b'])
#            z_mean = self.nonlinearity2(z_mean)
#            z_mean = tf.exp(z_mean)
            z_log_sigma_sq_a = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_encoder']['W5_to_log_sigma_a']), self.weights['VICI_encoder']['b5_to_log_sigma_a'])
            z_log_sigma_sq_b = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_encoder']['W5_to_log_sigma_b']), self.weights['VICI_encoder']['b5_to_log_sigma_b'])
            ab = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_encoder']['W4_to_ab']), self.weights['VICI_encoder']['b4_to_ab'])
#            z_log_sigma_sq = self.nonlinearity(z_log_sigma_sq+3)-3
            tf.summary.histogram("z_mean_a", z_mean_a)
            tf.summary.histogram("z_mean_b", z_mean_b)
            tf.summary.histogram("z_log_sigma_sq_a", z_log_sigma_sq_a)
            tf.summary.histogram("z_log_sigma_sq_b", z_log_sigma_sq_b)
            return z_mean_a, z_log_sigma_sq_a, z_mean_b, z_log_sigma_sq_b, ab

    def _sample_from_gaussian_dist(self, num_rows, num_cols, mean_a, log_sigma_sq_a, mean_b, log_sigma_sq_b, ab):
        with tf.name_scope("sample_in_z_space"):
            #eps_a = tf.random_normal([num_rows, num_cols], 0, 1., dtype=tf.float32)
            #eps_b = tf.random_normal([num_rows, num_cols], 0, 1., dtype=tf.float32)
            #sample_a = tf.add(mean_a, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq_a)), eps_a))
            #sample_b = tf.add(mean_b, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq_b)), eps_b))
            #idx = tf.random.categorical(ab, 1) #tf.dtypes.cast(num_rows,dtype=tf.int32))
            #matrix = tf.reshape(tf.tile(idx, [num_cols]), [num_rows, num_cols])
            #samples = tf.add(tf.mul(sample_a,matrix),tf.mul(sample_b,(1.0-matrix)))
            #bimix_gauss = tfd.Mixture(
            #              cat=tfd.Categorical(probs=[0.5*tf.ones([num_rows,num_cols]),0.5*tf.ones([num_rows,num_cols])]),
            #              components=[
            #                  tfd.Normal(loc=mean_a, scale=tf.sqrt(tf.exp(log_sigma_sq_a))),
            #                  tfd.Normal(loc=mean_b, scale=tf.sqrt(tf.exp(log_sigma_sq_b))),
            #              ])
            bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=ab),
                          components_distribution=tfd.MultivariateNormalDiag(
                              loc=tf.stack([mean_a,mean_b],axis=1),
                              scale_diag=tf.stack([tf.sqrt(tf.exp(log_sigma_sq_a)),tf.sqrt(tf.exp(log_sigma_sq_b))],axis=1)))
            sample = bimix_gauss.sample() #sample_shape=num_rows)
        return sample
    
#    def _sample_from_gaussian_conditional_dist(self, num_rows, num_cols, mean, x_multi, log_sigma_sq):
#        with tf.name_scope("sample_from_x_distribution"):
#            dp = tf.zeros([tf.shape(mean)[0],1],tf.float32)
##            samplef.stack([tf.sqrt(tf.exp(log_sigma_ = tf.zeros([tf.shape(mean)[0],tf.shape(mean)[1]],tf.float32)
#            sample = tf.zeros([tf.shape(mean)[0],0],dtype=tf.float32)
#            eps = tf.random_normal([num_rows, num_cols], 0, 1., dtype=tf.float32)
#            for i in range(num_cols):
#                mean_i = tf.expand_dims(mean[:,i],axis=1)
#                log_sigma_sq_i = tf.expand_dims(log_sigma_sq[:,i],axis=1)
#                eps_i = tf.expand_dims(eps[:,i],axis=1)
#                x_multi_i = tf.expand_dims(x_multi[:,i],axis=1)
##                eps = tf.random_normal([num_rows,1], 0, 1., dtype=tf.float32)
#                dp = tf.add(mean_i+tf.multiply(dp,x_multi_i), tf.multiply(tf.sqrt(tf.exp(log_sigma_sq_i)), eps_i))
##                sample[:,i] = dp
#                sample = tf.concat([sample,dp],1)
#                
#        return sample

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VICI_ENC"):
            # Encoder
            all_weights['VICI_encoder'] = collections.OrderedDict()
            hidden_number_encoder = self.n_weights
            all_weights['VICI_encoder']['W3_to_hidden'] = tf.Variable(vae_utils.xavier_init(self.n_input, hidden_number_encoder), dtype=tf.float32)
            tf.summary.histogram("W3_to_hidden", all_weights['VICI_encoder']['W3_to_hidden'])
    
            all_weights['VICI_encoder']['W3_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
            tf.summary.histogram("W3_hth", all_weights['VICI_encoder']['W3_hth'])
            
            all_weights['VICI_encoder']['W3b_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
            tf.summary.histogram("W3b_hth", all_weights['VICI_encoder']['W3b_hth'])
##    #        
#            all_weights['IVA_encoder']['W3c_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
#            tf.summary.histogram("W3c_hth", all_weights['IVA_encoder']['W3c_hth'])
    #        
    #        all_weights['encoder']['W3d_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
    #        tf.summary.histogram("W3d_hth", all_weights['encoder']['W3d_hth'])
    
            all_weights['VICI_encoder']['W4_to_mu_a'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, self.n_hidden),dtype=tf.float32)
            tf.summary.histogram("W4_to_mu_a", all_weights['VICI_encoder']['W4_to_mu_a'])
            all_weights['VICI_encoder']['W4_to_mu_b'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, self.n_hidden),dtype=tf.float32)
            tf.summary.histogram("W4_to_mu_b", all_weights['VICI_encoder']['W4_to_mu_b'])

            all_weights['VICI_encoder']['W5_to_log_sigma_a'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, self.n_hidden), dtype=tf.float32)
            tf.summary.histogram("W5_to_log_sigma_a", all_weights['VICI_encoder']['W5_to_log_sigma_a'])
            all_weights['VICI_encoder']['W5_to_log_sigma_b'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, self.n_hidden), dtype=tf.float32)
            tf.summary.histogram("W5_to_log_sigma_b", all_weights['VICI_encoder']['W5_to_log_sigma_b'])    
            all_weights['VICI_encoder']['W4_to_ab'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, 2), dtype=tf.float32)
            tf.summary.histogram("W4_to_ab", all_weights['VICI_encoder']['W4_to_ab'])

            all_weights['VICI_encoder']['b3_to_hidden'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b3_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b3b_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b3c_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b3d_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b4_to_mu_a'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
            all_weights['VICI_encoder']['b5_to_log_sigma_a'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
            all_weights['VICI_encoder']['b4_to_mu_b'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
            all_weights['VICI_encoder']['b5_to_log_sigma_b'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32) * self.bias_start, dtype=tf.float32)            
            all_weights['VICI_encoder']['b4_to_ab'] = tf.Variable(tf.zeros([2], dtype=tf.float32) * self.bias_start, dtype=tf.float32)

            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
