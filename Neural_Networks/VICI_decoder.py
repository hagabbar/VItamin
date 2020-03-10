import collections

import tensorflow as tf
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

class VariationalAutoencoder(object):

    def __init__(self, name, wrap_mask, nowrap_mask, n_input1=4, n_input2=256, n_output=3, n_weights=2048, n_hlayers=2, drate=0.2, n_filters=8, filter_size=8, maxpool=4, n_conv=2):
        
        self.n_input1 = n_input1                    # actually the output size
        self.n_input2 = n_input2                    # actually the output size
        self.n_output = n_output                  # the input data size
        self.n_weights = n_weights                # the number of weights were layer
        self.n_hlayers = n_hlayers
        self.n_conv = n_conv
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.maxpool = maxpool
        self.name = name                          # the name of the network
        self.drate = 0.2                          # dropout rate
        self.wrap_mask = wrap_mask                # mask identifying wrapped indices
        self.nowrap_mask = nowrap_mask            # mask identifying non-wrapped indices
 
        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinear_loc_nowrap = tf.sigmoid    # activation for non-wrapped location params
        self.nonlinear_loc_wrap = tf.sigmoid      # activation for wrapped location params
        self.nonlinear_scale_nowrap = tf.identity # activation for non-wrapped scale params
        self.nonlinear_scale_wrap = tf.nn.relu    # activation for wrapped scale params  
        self.nonlinearity = tf.nn.relu            # activation between hidden layers

    def calc_reconstruction(self, z, y):
        with tf.name_scope("VICI_decoder"):

            # Reshape input to a 3D tensor - single channel
            if self.n_conv is not None:
                conv_pool = tf.reshape(y, shape=[-1, 1, y.shape[1], 1])
                for i in range(self.n_conv):            
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    conv_pre = tf.add(tf.nn.conv2d(conv_pool, self.weights['VICI_decoder'][weight_name],strides=1,padding='SAME'),self.weights['VICI_decoder'][bias_name])
                    conv_post = self.nonlinearity(conv_pre)
                    conv_dropout = tf.layers.dropout(conv_post,rate=self.drate)
                    conv_pool = tf.nn.max_pool(conv_dropout,ksize=[1, 1, self.maxpool, 1],strides=[1, 1, self.maxpool, 1],padding='SAME')

                fc = tf.concat([z,tf.reshape(conv_pool, [-1, int(self.n_input2*self.n_filters/(self.maxpool**self.n_conv))])],axis=1)            

            else:
                fc = tf.concat([z,y],axis=1)

            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_decoder'][weight_name]), self.weights['VICI_decoder'][bias_name])
                hidden_post = self.nonlinearity(hidden_pre)
                hidden_dropout = tf.layers.dropout(hidden_post,rate=self.drate)
            loc_all = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_decoder']['w_loc']), self.weights['VICI_decoder']['b_loc'])
            scale_all = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_decoder']['w_scale']), self.weights['VICI_decoder']['b_scale'])

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
            
            if self.n_conv is not None:
                dummy = 1
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    all_weights['VICI_decoder'][weight_name] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, dummy*self.n_filters),[self.filter_size, 1, dummy, self.n_filters]), dtype=tf.float32)
                    all_weights['VICI_decoder'][bias_name] = tf.Variable(tf.zeros([self.n_filters], dtype=tf.float32))
                    tf.summary.histogram(weight_name, all_weights['VICI_decoder'][weight_name])
                    tf.summary.histogram(bias_name, all_weights['VICI_decoder'][bias_name])
                    dummy = self.n_filters

                fc_input_size = self.n_input1 + int(self.n_input2*self.n_filters/(self.maxpool**self.n_conv))
            else:
                fc_input_size = self.n_input1 + self.n_input2

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                all_weights['VICI_decoder'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights), dtype=tf.float32)
                all_weights['VICI_decoder'][bias_name] = tf.Variable(tf.zeros([self.n_weights], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VICI_decoder'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VICI_decoder'][bias_name])
                fc_input_size = self.n_weights
            all_weights['VICI_decoder']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights, self.n_output),dtype=tf.float32)
            all_weights['VICI_decoder']['b_loc'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_loc', all_weights['VICI_decoder']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VICI_decoder']['b_loc'])
            all_weights['VICI_decoder']['w_scale'] = tf.Variable(vae_utils.xavier_init(self.n_weights, self.n_output),dtype=tf.float32)
            all_weights['VICI_decoder']['b_scale'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_scale', all_weights['VICI_decoder']['w_scale'])
            tf.summary.histogram('b_scale', all_weights['VICI_decoder']['b_scale'])
            
            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
