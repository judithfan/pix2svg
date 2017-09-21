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
import torchvision.transforms as transforms
from transformtest import load_checkpoint
from generators import *


class TransformerLayerLossTest(BaseLossTest):
    def __init__(self, transformer_path, use_cuda=False):
        super(TransformerLayerLossTest, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        cnn = copy.deepcopy(vgg19.features)
        classifier = copy.deepcopy(vgg19.classifier)
        transformer = load_checkpoint(transformer_path, use_cuda=use_cuda)
        
        cnn.eval()
        classifier.eval()
        transformer.eval()

        if use_cuda:
            cnn.cuda()
            classifier.cuda()
            transformer.cuda()
        
        self.cnn = cnn
        self.classifier = classifier
        self.transformer = transformer
        self.use_cuda = use_cuda

    def loss(self, images, sketches):
        images_emb = self.cnn(images)
        sketches_emb = self.cnn(sketches)
        images_emb = images_emb.view(images_emb.size(0), -1)        
        sketches_emb = sketches_emb.view(sketches_emb.size(0), -1)

        layers = list(self.classifier)
        n_layers = len(layers)

        fc_i = 1
        for i in range(n_layers):
            if isinstance(layers[i], nn.Linear):
                name = 'fc_{index}'.format(index=fc_i)
                fc_i += 1

            images_emb = layers[i](images_emb)
            sketches_emb = layers[i](sketches_emb)

            if name == 'fc_2':
                return gen_distance(self.transformer(images_emb), sketches_emb, 
                                    metric='euclidean')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('transformer_path', type=str)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--datatype', type=str, default='data')
    args = parser.parse_args()

    assert args.datatype in ['data', 'noisy', 'swapped']
    print('-------------------------')
    print('Batch Size: {}'.format(args.batch))
    print('Out Directory: {}'.format(args.outdir))
    print('Data Type: {}'.format(args.datatype))
    print('Using Classifier: {}'.format(args.classifier))
    print('-------------------------')
    print('')

    use_cuda = torch.cuda.is_available()
    if args.datatype == 'data':
        generator = data_generator(imsize=224, use_cuda=use_cuda) # distance between sketch and target photo
    elif args.datatype == 'noisy':
        generator = noisy_generator(imsize=224, use_cuda=use_cuda) 
    elif args.datatype == 'swapped':
        generator = swapped_generator(imsize=224, use_cuda=use_cuda) # distance between sketch and photo from different class

    layer_test = TransformerLayerLossTest(args.transformer_path, use_cuda=use_cuda)

    b = 0  # number of batches
    n = 0  # number of examples
    quit = False
    loss_list = []

    if generator:
        while True:
            photo_batch = Variable(torch.zeros(args.batch, 3, 224, 224), volatile=True)
            sketch_batch = Variable(torch.zeros(args.batch, 3, 224, 224), volatile=True)
  
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

            losses = layer_test.loss(photo_batch, sketch_batch)
            losses = losses.cpu().data.numpy().flatten()
            loss_list += losses.tolist()
            
            n += (b + 1)

            if quit: 
                break

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    loss_list = np.array(loss_list)
    filename = 'loss_transformer_{datatype}.npy'.format(datatype=args.datatype)
    np.save(os.path.join(args.outdir, filename), loss_list)
