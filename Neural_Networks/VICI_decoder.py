import collections

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math as m

from Neural_Networks import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

class VariationalAutoencoder(object):

    def __init__(self, name, wrap_mask, nowrap_mask, n_input1=4, n_input2=256, n_output=3, n_weights=2048, n_hlayers=2, drate=0.2, n_filters=8, filter_size=8, maxpool=4, n_conv=2, conv_strides=1, pool_strides=1, num_det=1, batch_norm=False,by_channel=False, weight_init='xavier'):
        
        self.n_input1 = n_input1                    # actually the output size
        self.n_input2 = n_input2                    # actually the output size
        self.n_output = n_output                  # the input data size
        self.n_weights = n_weights                # the number of weights were layer
        self.n_hlayers = n_hlayers
        self.n_conv = n_conv
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.maxpool = maxpool
        self.conv_strides = conv_strides
        self.pool_strides = pool_strides
        self.name = name                          # the name of the network
        self.drate = drate                        # dropout rate
        self.wrap_mask = wrap_mask                # mask identifying wrapped indices
        self.nowrap_mask = nowrap_mask            # mask identifying non-wrapped indices
        self.num_det = num_det
        self.batch_norm = batch_norm
        self.by_channel = by_channel
        self.weight_init = weight_init 

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
                if self.by_channel == True:
                    conv_pool = tf.reshape(y, shape=[-1, 1, y.shape[1], self.num_det])
                    for i in range(self.n_conv):            
                        weight_name = 'w_conv_' + str(i)
                        bias_name = 'b_conv_' + str(i)
                        conv_pre = tf.add(tf.nn.conv2d(conv_pool, self.weights['VICI_decoder'][weight_name],strides=[1,1,self.conv_strides[i],1],padding='SAME'),self.weights['VICI_decoder'][bias_name])
                        conv_post = self.nonlinearity(conv_pre)
                        if self.batch_norm == True:
                            conv_batchNorm = tf.nn.batch_normalization(conv_post,tf.Variable(tf.zeros([1,conv_post.shape[2],conv_post.shape[3]], dtype=tf.float32)),tf.Variable(tf.ones([1,conv_post.shape[2],conv_post.shape[3]], dtype=tf.float32)),None,None,0.000001)
                            conv_dropout = tf.layers.dropout(conv_batchNorm,rate=self.drate)
                        else:
                            conv_dropout = tf.layers.dropout(conv_post,rate=self.drate)
                        conv_pool = tf.nn.max_pool(conv_dropout,ksize=[1, 1, self.maxpool[i], 1],strides=[1, 1, self.pool_strides[i], 1],padding='SAME')

                    fc = tf.concat([z,tf.reshape(conv_pool, [-1, int(conv_pool.shape[2]*conv_pool.shape[3])])],axis=1)            
                if self.by_channel == False:
                    conv_pool = tf.reshape(y, shape=[-1, y.shape[1], y.shape[2], 1])
                    for i in range(self.n_conv):
                        weight_name = 'w_conv_' + str(i)
                        bias_name = 'b_conv_' + str(i)
                        conv_pre = tf.add(tf.nn.conv2d(conv_pool, self.weights['VICI_decoder'][weight_name],strides=[1,self.conv_strides[i],self.conv_strides[i],1],padding='SAME'),self.weights['VICI_decoder'][bias_name])
                        conv_post = self.nonlinearity(conv_pre)
                        if self.batch_norm == True:
                            conv_batchNorm = tf.nn.batch_normalization(conv_post,tf.Variable(tf.zeros([conv_post.shape[1],conv_post.shape[2],conv_post.shape[3]], dtype=tf.float32)),tf.Variable(tf.ones([conv_post.shape[1],conv_post.shape[2],conv_post.shape[3]], dtype=tf.float32)),None,None,0.000001)
                        conv_pool = tf.nn.max_pool(conv_batchNorm,ksize=[1, self.maxpool[i], self.maxpool[i], 1],strides=[1, self.pool_strides[i], self.pool_strides[i], 1],padding='SAME')

                    fc = tf.concat([z,tf.reshape(conv_pool, [-1, int(conv_pool.shape[1]*conv_pool.shape[2]*conv_pool.shape[3])])],axis=1)
            else:
                fc = tf.concat([z,y],axis=1)

            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_decoder'][weight_name]), self.weights['VICI_decoder'][bias_name])
                hidden_post = self.nonlinearity(hidden_pre)
                if self.batch_norm == True:
                    hidden_batchNorm = tf.nn.batch_normalization(hidden_post,tf.Variable(tf.zeros([hidden_post.shape[1]], dtype=tf.float32)),tf.Variable(tf.ones([hidden_post.shape[1]], dtype=tf.float32)),None,None,0.000001)
                    hidden_dropout = tf.layers.dropout(hidden_batchNorm,rate=self.drate)
                else:
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
                    # orthogonal init
                    if self.weight_init == 'Orthogonal':
                        shape_init = (self.filter_size[i],dummy*self.n_filters[i])
                        initializer = tf.keras.initializers.Orthogonal()
                        all_weights['VICI_decoder'][weight_name] = tf.Variable(tf.reshape(initializer(shape=shape_init),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    # Variance scaling
                    if self.weight_init == 'VarianceScaling':
                        shape_init = (self.filter_size[i],dummy*self.n_filters[i])
                        initializer = tf.keras.initializers.VarianceScaling()
                        all_weights['VICI_decoder'][weight_name] = tf.Variable(tf.reshape(initializer(shape=shape_init),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    # xavier initilization
                    if self.weight_init == 'xavier':
                        all_weights['VICI_decoder'][weight_name] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], dummy*self.n_filters[i]),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    all_weights['VICI_decoder'][bias_name] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    tf.summary.histogram(weight_name, all_weights['VICI_decoder'][weight_name])
                    tf.summary.histogram(bias_name, all_weights['VICI_decoder'][bias_name])
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
                    fc_input_size = self.n_input1 + int(self.n_input2*self.n_filters[i]/(2**total_pool_stride_sum))
                else:
                    fc_input_size = self.n_input1 + int(self.n_input2*self.n_filters[i]/(2**total_pool_stride_sum)*2)
            else:
                fc_input_size = self.n_input1 + self.n_input2

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                all_weights['VICI_decoder'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights[i]), dtype=tf.float32)
                all_weights['VICI_decoder'][bias_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VICI_decoder'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VICI_decoder'][bias_name])
                fc_input_size = self.n_weights[i]
            all_weights['VICI_decoder']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output),dtype=tf.float32)
            all_weights['VICI_decoder']['b_loc'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_loc', all_weights['VICI_decoder']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VICI_decoder']['b_loc'])
            all_weights['VICI_decoder']['w_scale'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output),dtype=tf.float32)
            all_weights['VICI_decoder']['b_scale'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_scale', all_weights['VICI_decoder']['w_scale'])
            tf.summary.histogram('b_scale', all_weights['VICI_decoder']['b_scale'])
            
            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
