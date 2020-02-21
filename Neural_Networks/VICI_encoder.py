import collections

import tensorflow as tf
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

class VariationalAutoencoder(object):

    def __init__(self, name, n_input, n_hidden, n_weights, n_modes, middle="gaussian"):
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_weights = n_weights
        self.n_modes = n_modes
        self.name = name
        self.middle = middle
        self.bias_start = 0.0
        self.drate = 0.0
        self.mean_min = -10.0
        self.mean_max = 10.0
        self.log_sig_sq_min = -10.0
        self.log_sig_sq_max = 5.0
        self.log_weight_min = -10.0
        self.log_weight_max = 0.0

        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinearity = tf.nn.relu
        self.nonlinearity_mean = tf.clip_by_value
        #self.nonlinearity_log_sig_sq = tf.clip_by_value
        #self.nonlinearity_log_weight = tf.clip_by_value        


        #self.nonlinearity = tf.nn.leaky_relu


    def _calc_z_mean_and_sigma(self,x):
        with tf.name_scope("VICI_encoder"):
            hidden1_pre = tf.add(tf.matmul(x, self.weights['VICI_encoder']['W3_to_hidden']), self.weights['VICI_encoder']['b3_to_hidden'])
            hidden1_post = self.nonlinearity(hidden1_pre)
            hidden1_dropout = tf.layers.dropout(hidden1_post,rate=self.drate)

            hidden2_pre = tf.add(tf.matmul(hidden1_dropout, self.weights['VICI_encoder']['W3_hth']), self.weights['VICI_encoder']['b3_hth'])
            hidden2_post = self.nonlinearity(hidden2_pre)
            hidden2_dropout = tf.layers.dropout(hidden2_post,rate=self.drate)

            hidden3_pre = tf.add(tf.matmul(hidden2_dropout, self.weights['VICI_encoder']['W3b_hth']), self.weights['VICI_encoder']['b3b_hth'])
            hidden3_post = self.nonlinearity(hidden3_pre)
            hidden3_dropout = tf.layers.dropout(hidden3_post,rate=self.drate)

            z_mean = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_encoder']['W4_to_mu']), self.weights['VICI_encoder']['b4_to_mu'])
            #z_mean = self.nonlinearity_mean(z_mean,self.mean_min,self.mean_max) # clip the mean output
            #z_log_sig_sq = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_encoder']['W5_to_log_sigma']), self.weights['VICI_encoder']['b5_to_log_sigma'])
            #z_log_sig_sq_clipped = self.nonlinearity_log_sig_sq(z_log_sig_sq,self.log_sig_sq_min,self.log_sig_sq_max) # clip the mean output           
            #z_log_weight = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_encoder']['W5_to_weight']), self.weights['VICI_encoder']['b5_to_weight']) 
            #z_log_weight_clipped = self.nonlinearity_log_weight(z_log_weight,self.log_weight_min,self.log_weight_max) # clip the mean output

            tf.summary.histogram("z_mean", z_mean)
            #tf.summary.histogram("z_log_sigma_sq", z_log_sig_sq)
            #tf.summary.histogram("z_log_weight", z_log_weight)            

            #return z_mean, z_log_sig_sq, z_log_weight
            return tf.reshape(z_mean,(-1,self.n_modes,self.n_hidden))    

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VICI_ENC"):
            
            all_weights['VICI_encoder'] = collections.OrderedDict()
            hidden_number_encoder = self.n_weights
            
            # weights
            all_weights['VICI_encoder']['W3_to_hidden'] = tf.Variable(vae_utils.xavier_init(self.n_input, hidden_number_encoder), dtype=tf.float32)
            tf.summary.histogram("W3_to_hidden", all_weights['VICI_encoder']['W3_to_hidden'])
    
            all_weights['VICI_encoder']['W3_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
            tf.summary.histogram("W3_hth", all_weights['VICI_encoder']['W3_hth'])
            
            all_weights['VICI_encoder']['W3b_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
            tf.summary.histogram("W3b_hth", all_weights['VICI_encoder']['W3b_hth'])
    
            all_weights['VICI_encoder']['W4_to_mu'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, self.n_hidden*self.n_modes),dtype=tf.float32)
            tf.summary.histogram("W4_to_mu", all_weights['VICI_encoder']['W4_to_mu'])

            #all_weights['VICI_encoder']['W5_to_log_sigma'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, self.n_hidden), dtype=tf.float32)
            #tf.summary.histogram("W5_to_log_sigma", all_weights['VICI_encoder']['W5_to_log_sigma'])

            #all_weights['VICI_encoder']['W5_to_weight'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, 1), dtype=tf.float32)
            #tf.summary.histogram("W5_to_weight", all_weights['VICI_encoder']['W5_to_weight'])

            # biases
            all_weights['VICI_encoder']['b3_to_hidden'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b3_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b3b_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b3c_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b3d_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
            all_weights['VICI_encoder']['b4_to_mu'] = tf.Variable(tf.zeros([self.n_hidden*self.n_modes], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
            #all_weights['VICI_encoder']['b5_to_log_sigma'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
            #all_weights['VICI_encoder']['b5_to_weight'] = tf.Variable(tf.zeros([1], dtype=tf.float32) * self.bias_start, dtype=tf.float32)

            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
