""" pix2svg

Building variational autoencoder that takes photographs as input and outputs a sequence of strokes that form a figurative sketch

Encoder = pretrained VGG 
Decoder = MultiRNN LSTM 

References:	
	Sangkloy et al. 
	Ha & Eck sketch-rnn
	Simonyan & Zisserman VGG paper
	Kingma & Welling variational autoencdoer paper
	Graves RNN paper

Resources:
	Datasets:
		Sketchy dataset: photo-sketch pairs: http://sketchy.eye.gatech.edu/
	
	Encoders:	
		VGG model weights:  https://www.dropbox.com/s/9li9mi4105jf45v/vgg16.tflearn?dl=0
						https://github.com/tflearn/models	
						(https://www.cs.toronto.edu/~frossard/post/vgg16/)						
		FastMaskRCNN: https://github.com/CharlesShang/FastMaskRCNN
	
	Decoder:
		sketch-rnn: https://github.com/hardmaru/sketch-rnn

	Higher-level APIs for Tensorflow:
		TFLearn: https://github.com/tflearn/tflearn
		NeuroAILab tfutils: https://github.com/neuroailab/tfutils

"""
import numpy as np
import tensorflow as tf
import tflearn
import os
from model import Model


# define encoder
def vgg16(input, num_class):

    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
                                restore=False)
    return x

# define RNN decoder
# lives in sketch-rnn-model.py

# define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

host = os.uname()[1]
if host.startswith('node10'): # running on the cluster
	model_path = ../weights/vgg16.tflearn
else:
	print 'VGG weights are not stored locally on this machine.'


