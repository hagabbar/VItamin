import collections

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

class VariationalAutoencoder(object):

    def __init__(self, name, n_input=256, n_output=4, n_weights=2048, n_modes=2, n_hlayers=2, drate=0.2, n_filters=8, filter_size=8, maxpool=4, n_conv=2, conv_strides=1, pool_strides=1, num_det=1, batch_norm=False, by_channel=False, weight_init='xavier'):
        
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
        self.conv_strides = conv_strides
        self.pool_strides = pool_strides
        self.num_det = num_det
        self.batch_norm = batch_norm
        self.by_channel = by_channel
        self.weight_init = weight_init

        network_weights = self._create_weights()
        self.weights = network_weights


        self.nonlinearity = tf.nn.relu
        self.nonlinearity_mean = tf.clip_by_value

    def _calc_z_mean_and_sigma(self,x):
        with tf.name_scope("VICI_encoder"):
 
            # Reshape input to a 3D tensor - single channel
            if self.n_conv is not None:
                if self.by_channel == True:
                    conv_pool = tf.reshape(x, shape=[-1, 1, x.shape[1], self.num_det])

                    # network for messing with kernel size
                    for i in range(self.n_conv):
                        weight_name = 'w_conv_' + str(i)
                        bias_name = 'b_conv_' + str(i)
                        conv_pre = tf.add(tf.nn.conv2d(conv_pool, self.weights['VICI_encoder'][weight_name],strides=[1,1,self.conv_strides[i],1],padding='SAME'),self.weights['VICI_encoder'][bias_name])
                        conv_post = self.nonlinearity(conv_pre)
                        if self.batch_norm == True:
                            conv_batchNorm = tf.nn.batch_normalization(conv_post,tf.Variable(tf.zeros([1,conv_post.shape[2],conv_post.shape[3]], dtype=tf.float32)),tf.Variable(tf.ones([1,conv_post.shape[2],conv_post.shape[3]], dtype=tf.float32)),None,None,0.000001)
                            conv_dropout = tf.layers.dropout(conv_batchNorm,rate=self.drate)
                        else:
                            conv_dropout = tf.layers.dropout(conv_post,rate=self.drate)
                        conv_pool = tf.nn.max_pool(conv_dropout,ksize=[1, 1, self.maxpool[i], 1],strides=[1, 1, self.pool_strides[i], 1],padding='SAME')
                    
                    fc = tf.reshape(conv_pool, [-1, int(conv_pool.shape[2]*conv_pool.shape[3])])
                if self.by_channel == False:
                    conv_pool = tf.reshape(x, shape=[-1, x.shape[1], x.shape[2], 1])

                    # network for messing with kernel size
                    for i in range(self.n_conv):
                        weight_name = 'w_conv_' + str(i)
                        bias_name = 'b_conv_' + str(i)
                        conv_pre = tf.add(tf.nn.conv2d(conv_pool, self.weights['VICI_encoder'][weight_name],strides=[1,self.conv_strides[i],self.conv_strides[i],1],padding='SAME'),self.weights['VICI_encoder'][bias_name])
                        conv_post = self.nonlinearity(conv_pre)
                        if self.batch_norm == True:
                            conv_batchNorm = tf.nn.batch_normalization(conv_post,tf.Variable(tf.zeros([conv_post.shape[1],conv_post.shape[2],conv_post.shape[3]], dtype=tf.float32)),tf.Variable(tf.ones([conv_post.shape[1],conv_post.shape[2],conv_post.shape[3]], dtype=tf.float32)),None,None,0.000001)
                        conv_pool = tf.nn.max_pool(conv_batchNorm,ksize=[1, self.maxpool[i], self.maxpool[i], 1],strides=[1, self.pool_strides[i], self.pool_strides[i], 1],padding='SAME')

                    fc = tf.reshape(conv_pool, [-1, int(conv_pool.shape[1]*conv_pool.shape[2]*conv_pool.shape[3])])
            else:
                fc = x

            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_encoder'][weight_name]), self.weights['VICI_encoder'][bias_name])
                hidden_post = self.nonlinearity(hidden_pre)
                if self.batch_norm == True:
                    hidden_batchNorm = tf.nn.batch_normalization(hidden_post,tf.Variable(tf.zeros([hidden_post.shape[1]], dtype=tf.float32)),tf.Variable(tf.ones([hidden_post.shape[1]], dtype=tf.float32)),None,None,0.000001)
                    hidden_dropout = tf.layers.dropout(hidden_batchNorm,rate=self.drate)
                else:
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
                    # orthogonal init
                    if self.weight_init == 'Orthogonal':
                        shape_init = (self.filter_size[i],dummy*self.n_filters[i])
                        initializer = tf.keras.initializers.Orthogonal()
                        all_weights['VICI_encoder'][weight_name] = tf.Variable(tf.reshape(initializer(shape=shape_init),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    # Variance scaling
                    if self.weight_init == 'VarianceScaling':
                        shape_init = (self.filter_size[i],dummy*self.n_filters[i])
                        initializer = tf.keras.initializers.VarianceScaling()
                        all_weights['VICI_encoder'][weight_name] = tf.Variable(tf.reshape(initializer(shape=shape_init),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    # xavier initilization
                    if self.weight_init == 'xavier':
                        all_weights['VICI_encoder'][weight_name] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], dummy*self.n_filters[i]),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    all_weights['VICI_encoder'][bias_name] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    tf.summary.histogram(weight_name, all_weights['VICI_encoder'][weight_name])
                    tf.summary.histogram(bias_name, all_weights['VICI_encoder'][bias_name])
                    dummy = self.n_filters[i]

                total_pool_stride_sum = 0
                for j in range(len(self.maxpool)):
                    if self.maxpool[j] != 1 and self.pool_strides[j] != 1:
                        total_pool_stride_sum += 1
                    else:
                        if self.maxpool[j] != 1:
                            total_pool_stride_sum += 1
                        if self.pool_strides[j] != 1:
                            total_pool_stride_sum += 1
                    if self.conv_strides[j] != 1:
                        total_pool_stride_sum += 1
                if self.by_channel == True:
                    fc_input_size = int(self.n_input*self.n_filters[i]/(2**total_pool_stride_sum))
                else:
                    fc_input_size = int((self.n_input*self.n_filters[i]/(2**total_pool_stride_sum)*2))
            else:
                fc_input_size = self.n_input

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                all_weights['VICI_encoder'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights[i]), dtype=tf.float32)
                all_weights['VICI_encoder'][bias_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VICI_encoder'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VICI_encoder'][bias_name])
                fc_input_size = self.n_weights[i]
            all_weights['VICI_encoder']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output*self.n_modes),dtype=tf.float32)
            all_weights['VICI_encoder']['b_loc'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_loc', all_weights['VICI_encoder']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VICI_encoder']['b_loc'])
            all_weights['VICI_encoder']['w_scale'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output*self.n_modes),dtype=tf.float32)
            all_weights['VICI_encoder']['b_scale'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_scale', all_weights['VICI_encoder']['w_scale'])
            tf.summary.histogram('b_scale', all_weights['VICI_encoder']['b_scale'])
            all_weights['VICI_encoder']['w_weight'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_modes),dtype=tf.float32)
            all_weights['VICI_encoder']['b_weight'] = tf.Variable(tf.zeros([self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_weight', all_weights['VICI_encoder']['w_weight'])
            tf.summary.histogram('b_weight', all_weights['VICI_encoder']['b_weight'])

            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
