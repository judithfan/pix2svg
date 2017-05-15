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
	Sketchy dataset: photo-sketch pairs
	VGG model weights: https://www.cs.toronto.edu/~frossard/post/vgg16/
	TFLearn: https://github.com/tflearn/tflearn

"""

