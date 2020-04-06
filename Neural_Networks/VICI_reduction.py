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

    def __init__(self, name, n_input, filter_size, filter_channels, n_convsteps, red=False):
 
        self.n_convsteps = n_convsteps       
        self.n_input = n_input
        self.filter_size = filter_size
        self.filter_channels = filter_channels
        self.name = name
        self.red = red
        self.weights = self._create_weights()
        self.nonlinearity = tf.nn.relu
        self.outshape = [-1,int(n_input*filter_channels[0]/2**n_convsteps)] if red==False else [-1,int(n_input/2**n_convsteps)]

    def dimensionanily_reduction(self,x):
        with tf.name_scope(self.name):
            
            if self.n_convsteps==0:
                return x

#            redx_post0 = tf.expand_dims(x,2)
#            print(redx_post0.shape)
#            exit()
            redx_post0 = x
            if self.n_convsteps>=1:
                redxa_pre1 = tf.add(tf.nn.conv1d(redx_post0,self.weights[self.name]['F_conv_1a'],stride = 2, padding = 'SAME'),self.weights[self.name]['b_conv_1a'])
                redx_post1 = self.nonlinearity(redxa_pre1)
                redx_post = redx_post1
            
            if self.n_convsteps>=2:
                redxa_pre2 = tf.add(tf.nn.conv1d(redx_post1,self.weights[self.name]['F_conv_2a'],stride = 2, padding = 'SAME'),self.weights[self.name]['b_conv_2a'])
                redxb_pre2 = tf.nn.conv1d(redx_post0,self.weights[self.name]['F_conv_2b'],stride = 4, padding = 'SAME')
                redx_post2 = self.nonlinearity(redxa_pre2 + redxb_pre2)
                redx_post = redx_post2            

            if self.n_convsteps>=3:
                redxa_pre3 = tf.add(tf.nn.conv1d(redx_post2,self.weights[self.name]['F_conv_3a'],stride = 2, padding = 'SAME'),self.weights[self.name]['b_conv_3a'])
                redxb_pre3 = tf.nn.conv1d(redx_post1,self.weights[self.name]['F_conv_3b'],stride = 4, padding = 'SAME')
                redxc_pre3 = tf.nn.conv1d(redx_post0,self.weights[self.name]['F_conv_3c'],stride = 8, padding = 'SAME')
                redx_post3 = self.nonlinearity(redxa_pre3 + redxb_pre3 + redxc_pre3)
                redx_post = redx_post3            

            if self.n_convsteps>=4:
                redxa_pre4 = tf.add(tf.nn.conv1d(redx_post3,self.weights[self.name]['F_conv_4a'],stride = 2, padding = 'SAME'),self.weights[self.name]['b_conv_4a'])
                redxb_pre4 = tf.nn.conv1d(redx_post2,self.weights[self.name]['F_conv_4b'],stride = 4, padding = 'SAME')
                redxc_pre4 = tf.nn.conv1d(redx_post1,self.weights[self.name]['F_conv_4c'],stride = 8, padding = 'SAME')
                redxd_pre4 = tf.nn.conv1d(redx_post0,self.weights[self.name]['F_conv_4d'],stride = 16, padding = 'SAME')
                redx_post4 = self.nonlinearity(redxa_pre4 + redxb_pre4 + redxc_pre4 + redxd_pre4)
                redx_post = redx_post4

            if self.n_convsteps>=5:
                redxa_pre5 = tf.add(tf.nn.conv1d(redx_post4,self.weights[self.name]['F_conv_5a'],stride = 2, padding = 'SAME'),self.weights[self.name]['b_conv_5a'])
                redxb_pre5 = tf.nn.conv1d(redx_post3,self.weights[self.name]['F_conv_5b'],stride = 4, padding = 'SAME')
                redxc_pre5 = tf.nn.conv1d(redx_post2,self.weights[self.name]['F_conv_5c'],stride = 8, padding = 'SAME')
                redxd_pre5 = tf.nn.conv1d(redx_post1,self.weights[self.name]['F_conv_5d'],stride = 16, padding = 'SAME')
                redxe_pre5 = tf.nn.conv1d(redx_post0,self.weights[self.name]['F_conv_5e'],stride = 32, padding = 'SAME')
                redx_post5 = self.nonlinearity(redxa_pre5 + redxb_pre5 + redxc_pre5 + redxd_pre5 + redxe_pre5)
                redx_post = redx_post5

            if self.n_convsteps>=6:
                print('ERROR: cannot reduce beyond a factor or 2**5')
                exit(1)

            return tf.reshape(redx_post,self.outshape)
 
    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope(self.name):
            
            all_weights[self.name] = collections.OrderedDict()
            
            if self.n_convsteps>=1:
                out_channels = 1 if self.n_convsteps==1 and self.red==True else self.filter_channels
                all_weights[self.name]['F_conv_1a'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, out_channels),[self.filter_size, 1, out_channels]), dtype=tf.float32)
                all_weights[self.name]['b_conv_1a'] = tf.Variable(tf.zeros([tf.cast(tf.round(self.n_input/2),dtype=tf.int32),out_channels], dtype=tf.float32), dtype=tf.float32)
            if self.n_convsteps>=2:
                out_channels = 1 if self.n_convsteps==2 and self.red==True else self.filter_channels
                all_weights[self.name]['F_conv_2a'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_2b'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, out_channels),[self.filter_size, 1, out_channels]), dtype=tf.float32)
                all_weights[self.name]['b_conv_2a'] = tf.Variable(tf.zeros([tf.cast(tf.round(self.n_input/4),dtype=tf.int32),out_channels], dtype=tf.float32), dtype=tf.float32)
            if self.n_convsteps>=3:
                out_channels = 1 if self.n_convsteps==3 and self.red==True else self.filter_channels
                all_weights[self.name]['F_conv_3a'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_3b'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_3c'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, out_channels),[self.filter_size, 1, out_channels]), dtype=tf.float32)
                all_weights[self.name]['b_conv_3a'] = tf.Variable(tf.zeros([tf.cast(tf.round(self.n_input/8),dtype=tf.int32),out_channels], dtype=tf.float32), dtype=tf.float32)
            if self.n_convsteps>=4:
                out_channels = 1 if self.n_convsteps==4 and self.red==True else self.filter_channels
                all_weights[self.name]['F_conv_4a'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_4b'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_4c'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_4d'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, out_channels),[self.filter_size, 1,out_channels]), dtype=tf.float32)
                all_weights[self.name]['b_conv_4a'] = tf.Variable(tf.zeros([tf.cast(tf.round(self.n_input/16),dtype=tf.int32),out_channels], dtype=tf.float32), dtype=tf.float32)
            if self.n_convsteps>=5:        
                out_channels = 1 if self.n_convsteps==5 and self.red==True else self.filter_channels
                all_weights[self.name]['F_conv_5a'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_5b'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_5c'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_5d'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, self.filter_channels*out_channels),[self.filter_size, self.filter_channels, out_channels]), dtype=tf.float32)
                all_weights[self.name]['F_conv_5e'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size, out_channels),[self.filter_size, 1,out_channels]), dtype=tf.float32)
                all_weights[self.name]['b_conv_5a'] = tf.Variable(tf.zeros([tf.cast(tf.round(self.n_input/32),dtype=tf.int32),out_channels], dtype=tf.float32), dtype=tf.float32)
            if self.n_convsteps>=6:
                print('ERROR: cannot reduce beyond a factor or 2**5')
                exit(1)
              

        return all_weights

