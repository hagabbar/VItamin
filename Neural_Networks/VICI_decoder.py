import collections

import tensorflow as tf
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

class VariationalAutoencoder(object):

    def __init__(self, name, n_input, n_hidden, n_weights, wrap_mask, nowrap_mask):
        
        self.n_input = n_input                    # actually the output size
        self.n_hidden = n_hidden                  # the input data size
        self.n_weights = n_weights                # the number of weights were layer
        self.name = name                          # the name of the network
        self.bias_start = 0.0                     # some sort of initial bias value
        self.drate = 0.0                          # dropout rate
        self.wrap_mask = wrap_mask                # mask identifying wrapped indices
        self.nowrap_mask = nowrap_mask            # mask identifying non-wrapped indices
 
        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinear_loc_nowrap = tf.sigmoid    # activation for non-wrapped location params
        self.nonlinear_loc_wrap = tf.sigmoid      # activation for wrapped location params
        self.nonlinear_scale_nowrap = tf.identity # activation for non-wrapped scale params
        self.nonlinear_scale_wrap = tf.nn.relu    # activation for wrapped scale params  
        self.nonlinearity = tf.nn.relu            # activation between hidden layers

    def calc_reconstruction(self, z):
        with tf.name_scope("VICI_decoder"):

            # layer 1
            hidden1_pre = tf.add(tf.matmul(z,self.weights['VICI_decoder']['w_layer1']), self.weights['VICI_decoder']['b_layer1'])
            hidden1_post = self.nonlinearity(hidden1_pre)
            hidden1_dropout = tf.layers.dropout(hidden1_post,rate=self.drate)

            # layer 2
            hidden2_pre = tf.add(tf.matmul(hidden1_dropout, self.weights['VICI_decoder']['w_layer2']), self.weights['VICI_decoder']['b_layer2'])
            hidden2_post = self.nonlinearity(hidden2_pre)
            hidden2_dropout = tf.layers.dropout(hidden2_post,rate=self.drate)

            # layer 3
            hidden3_pre = tf.add(tf.matmul(hidden2_dropout, self.weights['VICI_decoder']['w_layer3']), self.weights['VICI_decoder']['b_layer3'])
            hidden3_post = self.nonlinearity(hidden3_pre)
            hidden3_dropout = tf.layers.dropout(hidden3_post,rate=self.drate)

            # output layer
            loc_all = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_decoder']['w_loc']), self.weights['VICI_decoder']['b_loc'])
            scale_all = tf.add(tf.matmul(hidden3_dropout, self.weights['VICI_decoder']['w_scale']), self.weights['VICI_decoder']['b_scale'])            

            # split up the output into non-wrapped and wrapped params and apply appropriate activation
            loc_nowrap = self.nonlinear_loc_nowrap(tf.boolean_mask(loc_all,self.nowrap_mask,axis=1))
            scale_nowrap = self.nonlinear_scale_nowrap(tf.boolean_mask(scale_all,self.nowrap_mask,axis=1))
            if np.sum(self.wrap_mask)>0:
                loc_wrap = self.nonlinear_loc_wrap(tf.boolean_mask(loc_all,self.wrap_mask,axis=1))
                scale_wrap = -1.0*self.nonlinear_scale_wrap(tf.boolean_mask(scale_all,self.wrap_mask,axis=1))
                return loc_nowrap, scale_nowrap, loc_wrap, scale_wrap
            else:
                return loc_nowrap, scale_nowrap

    def _create_weights(self):
        all_weights = collections.OrderedDict()

        # Decoder
        with tf.variable_scope("VICI_DEC"):
            all_weights['VICI_decoder'] = collections.OrderedDict()
            hidden_number_decoder = self.n_weights
            all_weights['VICI_decoder']['w_layer1'] = tf.Variable(vae_utils.xavier_init(self.n_hidden, hidden_number_decoder), dtype=tf.float32)
            all_weights['VICI_decoder']['b_layer1'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
                
            all_weights['VICI_decoder']['w_layer2'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
            all_weights['VICI_decoder']['b_layer2'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
                
            all_weights['VICI_decoder']['w_layer3'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
            all_weights['VICI_decoder']['b_layer3'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
    
            all_weights['VICI_decoder']['w_loc'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input), dtype=tf.float32)
            all_weights['VICI_decoder']['b_loc'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
            all_weights['VICI_decoder']['w_scale'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input), dtype=tf.float32)
            all_weights['VICI_decoder']['b_scale'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
            
            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
