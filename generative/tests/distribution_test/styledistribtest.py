"""Given a bunch of photos and a bunch match sketches, calculate distance
as matching conv_4_2 for photo vs sketch and matching the gram matrix of 
a bunch of layers for sketch and generated sketch.
"""


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import copy
from glob import glob
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from generators import *
from distribtest import BaseLossTest
from distribtest import gen_distance


class StyleTransferLossTest(BaseLossTest):
    def __init__(self, style_weight=1000, content_weight=1, diagonal_only=False, 
                 use_cuda=False):
        super(StyleTransferLossTest, self).__init__()
        cnn = models.vgg19(pretrained=True).features
        cnn.eval()
        gram = GramMatrix(diagonal_only=diagonal_only)

        if use_cuda:
            cnn = cnn.cuda()
            gram = gram.cuda()

        self.cnn = cnn
        self.gram = gram
        self.diagonal_only = diagonal_only
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.use_cuda = use_cuda

    def loss(self, images, sketches, style_image):
        layers = list(self.cnn)
        n_layers = len(layers)
        n = images.size(0)
        conv_i, pool_i = 1, 1

        style_image = style_image.expand_as(images)
        content_losses = Variable(torch.zeros(n))
        style_losses = Variable(torch.zeros(n))
        if self.use_cuda:
            content_losses = content_losses.cuda()
            style_losses = style_losses.cuda()

        for i in range(n_layers):
            if isinstance(layers[i], nn.Conv2d):
                name = 'conv_{group}_{index}'.format(group=pool_i, index=conv_i) 
                conv_i += 1
            elif isinstance(layers[i], nn.MaxPool2d):
                pool_i += 1
                conv_i = 1

            images = layers[i](images)
            sketches = layers[i](sketches)
            style_image = layers[i](style_image)

            if name in ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1']:
                sketches_gram = self.gram(sketches) * self.style_weight
                style_image_gram = self.gram(style_image) * self.style_weight
                losses = gen_distance(sketches_gram.view(n, -1), style_image_gram.view(n, -1), 
                                      metric='euclidean')
                style_losses = torch.add(style_losses, losses)
            elif name == 'conv_4_2':
                images_c = images.clone() * self.content_weight
                sketches_c = sketches.clone() * self.content_weight
                losses = gen_distance(images_c.view(n, -1), sketches_c.view(n, -1), 
                                      metric='euclidean')
                content_losses = torch.add(content_losses, losses)
       
        losses = content_losses + style_losses
        return losses


class GramMatrix(nn.Module):
    def __init__(self, diagonal_only=False):
        super(GramMatrix, self).__init__()
        self.diagonal_only = diagonal_only

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        G =  G.div(a * b * c * d)

        if self.diagonal_only:
            # if diagonal_only, then only keep diagonal...
            G = torch.diag(G)

        return G


if __name__ == '__main__':
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('style_image_path', type=str, help='path to style image')
    parser.add_argument('style_image_class', type=str, help='ignore all examples in dataset with same class')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--style_weight', type=float, default=1000.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--diagonal_only', action='store_true', default=False)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--datatype', type=str, default='data')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    assert args.datatype in ['data', 'noisy', 'swapped', 'perturbed', \
                             'neighbor', 'sketchspsc', 'sketchdpsc', \
                             'sketchdpdc', 'photodpsc', 'photodpdc']

    print('-------------------------')
    print('Batch Size: {}'.format(args.batch))
    print('Output Directory: {}'.format(args.outdir))
    print('Data Type: {}'.format(args.datatype))
    print('Style Image Path: {}'.format(args.style_image_path))
    print('Style Image Class: {}'.format(args.style_image_class))
    print('-------------------------')
    print('')

    use_cuda = torch.cuda.is_available() and args.cuda
    if args.datatype == 'data':
        generator = data_generator(imsize=224, ignore_class=args.style_image_class, use_cuda=use_cuda) # distance between sketch and target photo
    elif args.datatype == 'noisy':
        generator = noisy_generator(imsize=224, ignore_class=args.style_image_class, use_cuda=use_cuda) # distance between sketch and randomly generatored lines 
    elif args.datatype == 'swapped':
        generator = swapped_generator(imsize=224, ignore_class=args.style_image_class, use_cuda=use_cuda) # distance between sketch and photo from different class
    elif args.datatype == 'perturbed':
        generator = perturbed_generator(imsize=224, use_cuda=use_cuda)
    elif args.datatype == 'neighbor':
        generator = neighbor_generator(imsize=224, use_cuda=use_cuda) # distance between sketch and non-target photo from same class
    elif args.datatype == 'sketchspsc':
        generator = sketchspsc_generator(imsize=224, use_cuda=use_cuda) # distance between two sketches of same photo
    elif args.datatype == 'sketchdpsc':
        generator = sketchdpsc_generator(imsize=224, use_cuda=use_cuda) # distance between two sketches of different photos from same class
    elif args.datatype == 'sketchdpdc':
        generator = sketchdpdc_generator(imsize=224, use_cuda=use_cuda) # distance between two sketches of different photos from different classes
    elif args.datatype == 'photodpsc':
        generator = photodpsc_generator(imsize=224, use_cuda=use_cuda) # distance between two photos in same class
    elif args.datatype == 'photodpdc':
        generator = photodpdc_generator(imsize=224, use_cuda=use_cuda) # distance between two photos in different classes           

    style_image = load_image(args.style_image_path, imsize=224, use_cuda=use_cuda)
    LayerTest = StyleTransferLossTest(style_weight=args.style_weight, content_weight=args.content_weight, 
                                      diagonal_only=args.diagonal_only, use_cuda=use_cuda)

    b = 0  # number of batches
    n = 0  # number of examples
    quit = False
    loss_list = []

    if generator:
        while True:
            photo_batch = Variable(torch.zeros(args.batch, 3, 224, 224))
            sketch_batch = Variable(torch.zeros(args.batch, 3, 224, 224))
  
            if use_cuda:
                photo_batch = photo_batch.cuda()
                sketch_batch = sketch_batch.cuda()

            print('Batch {} | Examples {}'.format(b + 1, n))
            for b in range(args.batch):
                try:
                    photo, sketch = generator.next()
                    photo_batch[b] = photo
                    sketch_batch[b] = sketch
                except StopIteration:
                    quit = True
                    break

            photo_batch = photo_batch[:b + 1]
            sketch_batch = sketch_batch[:b + 1]

            losses = LayerTest.loss(photo_batch, sketch_batch, style_image)
            losses = losses.cpu().data.numpy().flatten()
            loss_list += losses.tolist()
            
            n += (b + 1)

            if quit: 
                break

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    loss_list = np.array(loss_list)
    filename = 'loss_style_transfer_euclidean_{datatype}.npy'.format(datatype=args.datatype)
    np.save(os.path.join(args.outdir, filename), loss_list)
