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


class BaseLossTest(object):
    def __init__(self):
        super(BaseLossTest, self).__init__()

    def loss(self, images, sketches):
        pass

    def gen(self, images, sketches, minibatch=32):
        """Calculates distance between each image and sketch
        pair. We are primarily interested in the mean and std
        of the (image, sketch).

        :param images: torch Variable
        :param sketches: torch Variable 
        :return: torch Tensor of distances
        """
        n = images.size(0)
        n_minibatch = n // minibatch

        losses = np.zeros(n)

        for i in range(n_minibatch):
            i_s, i_e = i * minibatch, (i + 1) * minibatch
            loss_minibatch = self.loss(images[i_s:i_e], sketches[i_s:i_e])
            losses[i_s:i_e] = loss_minibatch.cpu().data.numpy()

        if n_minibatch * minibatch < n:
            i_s, i_e = n_minibatch * minibatch, n
            loss_minibatch = self.loss(images[i_s:n], sketches[i_s:n])
            losses[i_s:i_e] = loss_minibatch.cpu().data.numpy()

        return losses


class LinearLayerLossTest(BaseLossTest):
    def __init__(self, layer_name, distance='euclidean', use_cuda=False):
        super(LinearLayerLossTest, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        cnn = copy.deepcopy(vgg19.features)
        classifier = copy.deepcopy(vgg19.classifier)
        
        cnn.eval()
        for p in cnn.parameters():
            p.requires_grad = False

        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False

        if use_cuda:
            cnn = cnn.cuda()
            classifier = classifier.cuda()

        self.cnn = cnn
        self.classifier = classifier
        self.layer_name = layer_name
        self.distance = distance
        self.use_cuda = use_cuda

    def loss(self, images, sketches):
        images_emb = self.cnn(images)
        images_emb = images_emb.view(images_emb.size(0), -1)
        sketches_emb = self.cnn(sketches)
        sketches_emb = sketches_emb.view(sketches_emb.size(0), -1)

        layers = list(self.classifier)
        n_layers = len(layers)

        fc_i = 1
        relu_i = 1
        dropout_i = 1

        for i in range(n_layers):
            if isinstance(layers[i], nn.Linear):
                name = 'fc_{index}'.format(index=fc_i)
                fc_i += 1
            elif isinstance(layers[i], nn.ReLU):
                name = 'relu_{index}'.format(index=relu_i)
                relu_i += 1
            elif isinstance(layers[i], nn.Dropout):
                name = 'dropout_{index}'.format(index=dropout_i)
                dropout_i += 1
            else:
                raise Exception('layer {} not recognized'.format(type(layers[i])))

            images_emb = layers[i](images_emb)
            sketches_emb = layers[i](sketches_emb)

            if name == self.layer_name:
                return gen_distance(images_emb, sketches_emb, metric=self.distance)


class SingleLayerLossTest(BaseLossTest):
    def __init__(self, layer_name, distance='euclidean', use_cuda=False):
        super(SingleLayerLossTest, self).__init__()
        cnn = copy.deepcopy(models.vgg19(pretrained=True).features)
        cnn.eval()
        for p in cnn.parameters():
            p.requires_grad = False

        if use_cuda:
            cnn = cnn.cuda()
        self.cnn = cnn
        self.layer_name = layer_name
        self.distance = distance
        self.use_cuda = use_cuda

    def loss(self, images, sketches):
        layers = list(self.cnn)
        n_layers = len(layers)
        conv_i, relu_i, pool_i = 1, 1, 1

        for i in range(n_layers):
            if isinstance(layers[i], nn.Conv2d):
                name = 'conv_{group}_{index}'.format(group=pool_i, index=conv_i) 
                conv_i += 1
            elif isinstance(layers[i], nn.ReLU):
                name = 'relu_{group}_{index}'.format(group=pool_i, index=relu_i)
                relu_i += 1
            elif isinstance(layers[i], nn.MaxPool2d):
                name = 'pool_{index}'.format(index=pool_i)
                pool_i += 1
                conv_i = 1
                relu_i = 1
            else:
                raise Exception('layer {} not recognized'.format(type(layers[i])))

            images = layers[i](images)
            sketches = layers[i](sketches)

            if name == self.layer_name:
                a = images.view(images.size(0), -1)
                b = sketches.view(sketches.size(0), -1)
                return gen_distance(a, b, metric=self.distance)


class MultiLayerLossTest(BaseLossTest):
    def __init__(self, layer_name_list, distance='euclidean',
                 weight_list=None, use_cuda=False):
        super(SingleLayerLossTest, self).__init__()
        assert len(layer_name_list) > 0
        cnn = copy.deepcopy(models.vgg19(pretrained=True))
        cnn.eval()
        for p in cnn.parameters():
            p.requires_grad = False

        if use_cuda:
            cnn = cnn.cuda()

        if not weight_list:
            weight_list = [1 for i in range(len(layer_name_list))]

        layer_name_list.sort()

        self.cnn = cnn
        self.layer_name_list = layer_name_list
        self.distance = distance
        self.weight_list = weight_list
        self.use_cuda = use_cuda

    def loss(self, images, sketches):
        layers = list(self.cnn)
        n_layers = len(layers)
        conv_i, relu_i, pool_i = 1, 1, 1
        n = images.size(0)

        losses = Variable(torch.zeros(n))
        if self.use_cuda:
            losses = losses.cuda()

        for i in range(n_layers):
            if isinstance(layers[i], nn.Conv2d):
                name = 'conv_{group}_{index}'.format(group=pool_i, index=conv_i) 
                conv_i += 1
            elif isinstance(layers[i], nn.ReLU):
                name = 'relu_{group}_{index}'.format(group=pool_i, index=relu_i)
                relu_i += 1
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{index}'.format(index=pool_i)
                pool_i += 1
            else:
                raise Exception('layer {} not recognized'.format(type(layers[i])))

            images = layers[i](images)
            sketches = layers[i](sketches)

            if name in self.layer_name_list:
                a = images.view(images.size(0), -1)
                b = sketches.view(sketches.size(0), -1)
                layer_losses = gen_distance(a, b, metric=self.distance)
                layer_losses *= self.weight_list[self.layer_name_list.index(name)]
                losses = torch.add(losses, layer_losses)

        return losses


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.cosine_similarity(input1, input2)
        >>> print(output)
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def gen_distance(a, b, metric='cosine'):
    """Implementation of difference distance metrics ripped from Wolfram:
    http://reference.wolfram.com/language/guide/DistanceAndSimilarityMeasures.html
    """
    if metric == 'cosine':
        return 1 - cosine_similarity(a, b, dim=1)
    elif metric == 'euclidean':
        return torch.norm(a - b, p=2, dim=1)
    elif metric == 'squared_euclidean':
        return torch.pow(torch.norm(a - b, p=2, dim=1), 2)
    elif metric == 'normalized_squared_euclidean':
        c = a - torch.mean(a, dim=1).expand_as(a)
        d = b - torch.mean(b, dim=1).expand_as(b)
        n = torch.pow(torch.norm(c, p=2, dim=1), 2) + torch.pow(torch.norm(d, p=2, dim=1), 2)
        return 0.5 * torch.pow(torch.norm(c - d, p=2, dim=1), 2) / n
    elif metric == 'manhattan':
        return F.pairwise_distance(a, b, p=1)
    elif metric == 'chessboard':
        return torch.max(torch.abs(a - b), dim=1)[0]
    elif metric == 'bray_curtis':
        return torch.sum(torch.abs(a - b), dim=1) / torch.sum(torch.abs(a + b), dim=1)
    elif metric == 'canberra':
        return torch.sum(torch.abs(a - b) / (torch.abs(a) + torch.abs(b)), dim=1)
    elif metric == 'correlation':
        c = a - torch.mean(a, dim=1).expand_as(a)
        d = b - torch.mean(b, dim=1).expand_as(b)
        return 1 - cosine_similarity(c, d, dim=1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_name', type=str, default='conv_4_2')
    parser.add_argument('--distance', type=str, default='euclidean')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--datatype', type=str, default='data')
    parser.add_argument('--classifier', action='store_true', default=False)
    args = parser.parse_args()

    assert args.datatype in ['data', 'noisy', 'swapped', 'perturbed', \
                             'neighbor','sketchspsc','sketchdpsc','sketchdpdc','photodpsc','photodpdc']

    print('-------------------------')
    print('Layer Name: {}'.format(args.layer_name))
    print('Distance: {}'.format(args.distance))
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

    if args.classifier:
        layer_test = LinearLayerLossTest(args.layer_name, distance=args.distance, 
                                         use_cuda=use_cuda)
    else:
        layer_test = SingleLayerLossTest(args.layer_name, distance=args.distance, 
                                         use_cuda=use_cuda)

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
    filename = 'loss_{name}_{distance}_{datatype}.npy'.format(name=args.layer_name, distance=args.distance,datatype=args.datatype)
    np.save(os.path.join(args.outdir, filename), loss_list)