import collections

import tensorflow as tf
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

class VariationalAutoencoder(object):

    def __init__(self, name, n_hidden, filter_size, filter_channels, middle="gaussian"):
        
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.filter_channels = filter_channels
        self.name = name
        self.middle = middle
        self.bias_start = 0.0

        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinearity = tf.nn.relu
        self.nonlinearity2 = tf.nn.leaky_relu


    def dimensionanily_reduction(self,x):
        with tf.name_scope("VICI_encoder"):
           
            #X = tf.expand_dims(x,2)
            X = x
            hidden0_pre = tf.add(tf.compat.v1.nn.conv1d(X,self.weights[self.name]['F_conv_0'],stride = 2, padding = 'SAME'),self.weights[self.name]['b_conv_0'])
            hidden0_post = self.nonlinearity(hidden0_pre)
            
            hidden1_pre = tf.add(tf.nn.conv1d(hidden0_post,self.weights[self.name]['F_conv_1'],strides = 2, padding = 'SAME'),self.weights[self.name]['b_conv_1'])
            hidden1b_pre = tf.nn.conv1d(X,self.weights[self.name]['F_conv_1b'],stride = 4, padding = 'SAME')
            hidden1_post = self.nonlinearity(hidden1_pre+hidden1b_pre)
            hidden1_post = tf.squeeze(hidden1_post)
            
            redx_pre = tf.add(tf.nn.conv1d(hidden1_post,self.weights[self.name]['F_conv_2'],stride = 2, padding = 'SAME'),self.weights[self.name]['b_conv_2'])
            redxb_pre = tf.nn.conv1d(hidden0_post,self.weights[self.name]['F_conv_2b'],stride = 4, padding = 'SAME')
            redxc_pre = tf.nn.conv1d(X,self.weights[self.name]['F_conv_1c'],stride = 8, padding = 'SAME')
            redx_post = self.nonlinearity(redx_pre+redxb_pre+redxc_pre)
            
            redx_post = tf.squeeze(redx_post)

            return redx_post

    def _sample_from_gaussian_dist(self, num_rows, num_cols, mean, log_sigma_sq):
        with tf.name_scope("sample_in_z_space"):
            eps = tf.random_normal([num_rows, num_cols], 0, 1., dtype=tf.float32)
            sample = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps))
        return sample

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VICI_ENC"):
            # Encoder
            all_weights[self.name] = collections.OrderedDict()
            
            all_weights[self.name]['F_conv_0'] = tf.expand_dims(tf.Variable(vae_utils.xavier_init(self.filter_size, 1, self.filter_channels), dtype=tf.float32),0)
            all_weights[self.name]['b_conv_0'] = tf.Variable(tf.zeros([tf.cast(tf.round(self.n_hidden/2),dtype=tf.int32),self.filter_channels], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
            
            all_weights[self.name]['F_conv_1'] = tf.Variable(vae_utils.xavier_init(self.filter_size, self.filter_channels, self.filter_channels), dtype=tf.float32)
            all_weights[self.name]['F_conv_1b'] = tf.Variable(vae_utils.xavier_init(self.filter_size, 1, self.filter_channels), dtype=tf.float32)
            all_weights[self.name]['b_conv_1'] = tf.Variable(tf.zeros([tf.cast(tf.round(self.n_hidden/4),dtype=tf.int32),self.filter_channels], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
            
            all_weights[self.name]['F_conv_2'] = tf.Variable(vae_utils.xavier_init(self.filter_size, self.filter_channels, 1), dtype=tf.float32)
            all_weights[self.name]['F_conv_1b'] = tf.Variable(vae_utils.xavier_init(self.filter_size, self.filter_channels, self.filter_channels), dtype=tf.float32)
            all_weights[self.name]['F_conv_1c'] = tf.Variable(vae_utils.xavier_init(self.filter_size, 1, self.filter_channels), dtype=tf.float32)
            all_weights[self.name]['b_conv_1'] = tf.Variable(tf.zeros([tf.cast(tf.round(self.n_hidden/8),dtype=tf.int32),1], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
        
        return all_weights
