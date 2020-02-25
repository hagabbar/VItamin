import collections

import tensorflow as tf
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

class VariationalAutoencoder(object):

    def __init__(self, name, n_input=256, n_output=4, n_weights=2048, n_modes=2, n_hlayers=2, drate=0.2, n_filters=8, filter_size=8, maxpool=4, n_conv=2):
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_weights = n_weights
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_hlayers = n_hlayers
        self.n_conv = n_conv
        self.n_modes = n_modes
        self.drate = drate
        self.maxpool = maxpool

        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinearity = tf.nn.relu
        self.nonlinearity_mean = tf.clip_by_value

    def _calc_z_mean_and_sigma(self,x):
        with tf.name_scope("VICI_encoder"):
 
            # Reshape input to a 3D tensor - single channel
            if self.n_conv is not None:
                conv_pool = tf.reshape(x, shape=[-1, 1, x.shape[1], 1])
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    conv_pre = tf.add(tf.nn.conv2d(conv_pool, self.weights['VICI_encoder'][weight_name],strides=1,padding='SAME'),self.weights['VICI_encoder'][bias_name])
                    conv_post = self.nonlinearity(conv_pre)
                    conv_dropout = tf.layers.dropout(conv_post,rate=self.drate)
                    conv_pool = tf.nn.max_pool(conv_dropout,ksize=[1, 1, self.maxpool, 1],strides=[1, 1, self.maxpool, 1],padding='SAME')

                fc = tf.reshape(conv_pool, [-1, int(self.n_input*self.n_filters/(self.maxpool**self.n_conv))])

            else:
                fc = x
           
            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_encoder'][weight_name]), self.weights['VICI_encoder'][bias_name])
                hidden_post = self.nonlinearity(hidden_pre)
                hidden_dropout = tf.layers.dropout(hidden_post,rate=self.drate)
            loc = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_encoder']['w_loc']), self.weights['VICI_encoder']['b_loc'])
            scale = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_encoder']['w_scale']), self.weights['VICI_encoder']['b_scale'])
            weight = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_encoder']['w_weight']), self.weights['VICI_encoder']['b_weight']) 

            tf.summary.histogram('loc', loc)
            tf.summary.histogram('scale', scale)
            tf.summary.histogram('weight', weight)
            return tf.reshape(loc,(-1,self.n_modes,self.n_output)), tf.reshape(scale,(-1,self.n_modes,self.n_output)), tf.reshape(weight,(-1,self.n_modes))    

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VICI_ENC"):            
            all_weights['VICI_encoder'] = collections.OrderedDict()

            if self.n_conv is not None:
                dummy = 1
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    all_weights['VICI_encoder'][weight_name] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, dummy*self.n_filters),[self.filter_size, 1, dummy, self.n_filters]), dtype=tf.float32)
                    all_weights['VICI_encoder'][bias_name] = tf.Variable(tf.zeros([self.n_filters], dtype=tf.float32))
                    tf.summary.histogram(weight_name, all_weights['VICI_encoder'][weight_name])
                    tf.summary.histogram(bias_name, all_weights['VICI_encoder'][bias_name])
                    dummy = self.n_filters

            fc_input_size = int(self.n_input*self.n_filters/(self.maxpool**self.n_conv))
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                all_weights['VICI_encoder'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights), dtype=tf.float32)
                all_weights['VICI_encoder'][bias_name] = tf.Variable(tf.zeros([self.n_weights], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VICI_encoder'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VICI_encoder'][bias_name])
                fc_input_size = self.n_weights
            all_weights['VICI_encoder']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights, self.n_output*self.n_modes),dtype=tf.float32)
            all_weights['VICI_encoder']['b_loc'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_loc', all_weights['VICI_encoder']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VICI_encoder']['b_loc'])
            all_weights['VICI_encoder']['w_scale'] = tf.Variable(vae_utils.xavier_init(self.n_weights, self.n_output*self.n_modes),dtype=tf.float32)
            all_weights['VICI_encoder']['b_scale'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_scale', all_weights['VICI_encoder']['w_scale'])
            tf.summary.histogram('b_scale', all_weights['VICI_encoder']['b_scale'])
            all_weights['VICI_encoder']['w_weight'] = tf.Variable(vae_utils.xavier_init(self.n_weights, self.n_modes),dtype=tf.float32)
            all_weights['VICI_encoder']['b_weight'] = tf.Variable(tf.zeros([self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_weight', all_weights['VICI_encoder']['w_weight'])
            tf.summary.histogram('b_weight', all_weights['VICI_encoder']['b_weight'])

            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
