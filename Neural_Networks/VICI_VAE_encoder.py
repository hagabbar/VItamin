import collections

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

class VariationalAutoencoder(object):

    def __init__(self, name, n_input1=3, n_input2=256, n_output=4, n_weights=2048, n_hlayers=2, drate=0.2, n_filters=8, filter_size=8, maxpool=4, n_conv=2, strides=1, num_det=1):
        
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.n_output = n_output
        self.n_weights = n_weights

        self.n_hlayers = n_hlayers
        self.n_conv = n_conv
        self.drate = drate
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.maxpool = maxpool
        self.strides = strides
        self.num_det = num_det

        network_weights = self._create_weights()
        self.weights = network_weights
        self.nonlinearity = tf.nn.relu

    def _calc_z_mean_and_sigma(self,x,y):
        with tf.name_scope("VICI_VAE_encoder"):

            # Reshape input to a 3D tensor - single channel
            if self.n_conv is not None:
                conv_pool = tf.reshape(y, shape=[-1, 1, y.shape[1], self.num_det])
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    conv_pre = tf.add(tf.nn.conv2d(conv_pool, self.weights['VICI_VAE_encoder'][weight_name],strides=[1,1,self.strides,1],padding='SAME'),self.weights['VICI_VAE_encoder'][bias_name])
                    conv_post = self.nonlinearity(conv_pre)
                    conv_dropout = tf.layers.dropout(conv_post,rate=self.drate)
                    conv_pool = tf.nn.max_pool(conv_dropout,ksize=[1, 1, self.maxpool, 1],strides=[1, 1, self.maxpool, 1],padding='SAME')

                fc = tf.concat([x,tf.reshape(conv_pool, [-1, int(self.n_input2*self.n_filters/(self.maxpool**self.n_conv))])],axis=1)

            else:
                fc = tf.concat([x,y],axis=1)

            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_VAE_encoder'][weight_name]), self.weights['VICI_VAE_encoder'][bias_name])
                hidden_post = self.nonlinearity(hidden_pre)
                hidden_dropout = tf.layers.dropout(hidden_post,rate=self.drate)
            loc = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_VAE_encoder']['w_loc']), self.weights['VICI_VAE_encoder']['b_loc'])
            scale = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_VAE_encoder']['w_scale']), self.weights['VICI_VAE_encoder']['b_scale'])

            tf.summary.histogram('loc', loc)
            tf.summary.histogram('scale', scale)
            return loc, scale


    def _sample_from_gaussian_dist(self, num_rows, num_cols, mean, log_sigma_sq):
        with tf.name_scope("sample_in_z_space"):
            eps = tf.random_normal([num_rows, num_cols], 0, 1., dtype=tf.float32)
            sample = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps))
        return sample

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VICI_VAE_ENC"):
            # Encoder
            all_weights['VICI_VAE_encoder'] = collections.OrderedDict()
            
            if self.n_conv is not None:
                dummy = 1
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    all_weights['VICI_VAE_encoder'][weight_name] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, dummy*self.n_filters),[self.filter_size, 1, dummy, self.n_filters]), dtype=tf.float32)
                    all_weights['VICI_VAE_encoder'][bias_name] = tf.Variable(tf.zeros([self.n_filters], dtype=tf.float32))
                    tf.summary.histogram(weight_name, all_weights['VICI_VAE_encoder'][weight_name])
                    tf.summary.histogram(bias_name, all_weights['VICI_VAE_encoder'][bias_name])
                    dummy = self.n_filters

                fc_input_size = self.n_input1 + int(self.n_input2*self.n_filters/(self.maxpool**self.n_conv))
            else:
                fc_input_size = self.n_input1 + self.n_input2

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                all_weights['VICI_VAE_encoder'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights), dtype=tf.float32)
                all_weights['VICI_VAE_encoder'][bias_name] = tf.Variable(tf.zeros([self.n_weights], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VICI_VAE_encoder'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VICI_VAE_encoder'][bias_name])
                fc_input_size = self.n_weights
            all_weights['VICI_VAE_encoder']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights, self.n_output),dtype=tf.float32)
            all_weights['VICI_VAE_encoder']['b_loc'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_loc', all_weights['VICI_VAE_encoder']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VICI_VAE_encoder']['b_loc'])
            all_weights['VICI_VAE_encoder']['w_scale'] = tf.Variable(vae_utils.xavier_init(self.n_weights, self.n_output),dtype=tf.float32)
            all_weights['VICI_VAE_encoder']['b_scale'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_scale', all_weights['VICI_VAE_encoder']['w_scale'])
            tf.summary.histogram('b_scale', all_weights['VICI_VAE_encoder']['b_scale'])

            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
