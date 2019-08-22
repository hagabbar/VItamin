import collections

import tensorflow as tf
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

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

    def calc_reconstruction(self, z):
        with tf.name_scope("VICI_decoder"):
            if self.middle == "bernoulli":
                hidden1_pre = tf.add(tf.matmul(z, self.weights['ELBO_decoder']['W1_to_hidden']), self.weights['ELBO_decoder']['b1_to_hidden'])
                hidden1_post = self.nonlinearity(hidden1_pre)
                hidden1_dropout = tf.layers.dropout(hidden1_post,rate=0.5)
                
                hidden3_pre = tf.add(tf.matmul(hidden1_dropout, self.weights['ELBO_decoder']['W1c_htoh']), self.weights['ELBO_decoder']['b1c_htoh'])
                hidden3_post = self.nonlinearity(hidden3_pre)

                hidden2_pre = tf.add(tf.matmul(hidden3_post, self.weights['ELBO_decoder']['W1b_htoh']), self.weights['ELBO_decoder']['b1b_htoh'])
                hidden2_post = self.nonlinearity(hidden2_pre)

                y_pre = tf.add(tf.matmul(hidden2_post, self.weights['ELBO_decoder']['W2_to_y_pre']), self.weights['ELBO_decoder']['b2_to_y_pre'])
                y = tf.sigmoid(y_pre)
                return y
            elif self.middle == "gaussian":
                hidden1_pre = tf.add(tf.matmul(z,self.weights['VICI_decoder']['W3_to_hiddenG']), self.weights['VICI_decoder']['b3_to_hiddenG'])
                hidden1_post = self.nonlinearity(hidden1_pre)
                hidden1_dropout = tf.layers.dropout(hidden1_post,rate=self.drate)
#                hidden1_post = tf.nn.batch_normalization(hidden1_post,tf.Variable(tf.zeros([400], dtype=tf.float32)),tf.Variable(tf.ones([400], dtype=tf.float32)),None,None,0.000001,name="d_b_norm_1")

#                hidden1_pre_s = tf.add(tf.matmul(z,self.weights['decoder']['W3_to_hiddenGS']), self.weights['decoder']['b3_to_hiddenGS'])
#                hidden1_post_s = self.nonlinearity(hidden1_pre_s)

                hidden1b_pre = tf.add(tf.matmul(hidden1_dropout, self.weights['VICI_decoder']['W3b_to_hiddenG']), self.weights['VICI_decoder']['b3b_to_hiddenG'])
                hidden1b_post = self.nonlinearity(hidden1b_pre)
                hidden1b_dropout = tf.layers.dropout(hidden1b_post,rate=self.drate)

                hidden1c_pre = tf.add(tf.matmul(hidden1b_dropout, self.weights['VICI_decoder']['W3c_to_hiddenG']), self.weights['VICI_decoder']['b3c_to_hiddenG'])
                hidden1c_post = self.nonlinearity(hidden1c_pre)
                hidden1c_dropout = tf.layers.dropout(hidden1c_post,rate=self.drate)
##                
#                hidden1d_pre = tf.add(tf.matmul(hidden1c_post, self.weights['decoder']['W3d_to_hiddenG']), self.weights['decoder']['b3d_to_hiddenG'])
#                hidden1d_post = self.nonlinearity(hidden1d_pre)
                
#                hidden1e_pre = tf.add(tf.matmul(hidden1d_post, self.weights['decoder']['W3e_to_hiddenG']), self.weights['decoder']['b3e_to_hiddenG'])
#                hidden1e_post = self.nonlinearity(hidden1e_pre)
                
#                hidden1b_post = hidden1_post

                mu = tf.add(tf.matmul(hidden1c_dropout, self.weights['VICI_decoder']['W4_to_muG']), self.weights['VICI_decoder']['b4_to_muG'])
                mu = tf.sigmoid(mu)  # see paper
#                mu = tf.sigmoid(mu+0.5)-0.5  # see paper
                log_sigma_sq = tf.add(tf.matmul(hidden1c_dropout, self.weights['VICI_decoder']['W5_to_log_sigmaG']), self.weights['VICI_decoder']['b5_to_log_sigmaG'])
#                log_sigma_sq = self.nonlinearity(log_sigma_sq+20)-20
#                log_sigma_sq = self.nonlinearity(log_sigma_sq+6)-6
                return mu, log_sigma_sq
            else:
                RuntimeError(self.middle + " is not yet constructed for reconstruction")


    def _create_weights(self):
        all_weights = collections.OrderedDict()

        # Decoder
        with tf.variable_scope("VICI_DEC"):
            all_weights['VICI_decoder'] = collections.OrderedDict()
            if self.middle == "gaussian":
                hidden_number_decoder = self.n_weights
                all_weights['VICI_decoder']['W3_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(self.n_hidden, hidden_number_decoder), dtype=tf.float32)
                all_weights['VICI_decoder']['b3_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
                
    #            all_weights['decoder']['W3_to_hiddenGS'] = tf.Variable(vae_utils.xavier_init(self.n_hidden, hidden_number_decoder), dtype=tf.float32)
    #            all_weights['decoder']['b3_to_hiddenGS'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
                
                all_weights['VICI_decoder']['W3b_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
                all_weights['VICI_decoder']['b3b_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
    #            
                all_weights['VICI_decoder']['W3c_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
                all_weights['VICI_decoder']['b3c_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
    #####            
    #            all_weights['decoder']['W3d_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
    #            all_weights['decoder']['b3d_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
    #            
    #            all_weights['decoder']['W3e_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
    #            all_weights['decoder']['b3e_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
    
                all_weights['VICI_decoder']['W4_to_muG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input), dtype=tf.float32)
                all_weights['VICI_decoder']['b4_to_muG'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
                all_weights['VICI_decoder']['W5_to_log_sigmaG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input), dtype=tf.float32)
                all_weights['VICI_decoder']['b5_to_log_sigmaG'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
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
