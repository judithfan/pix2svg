from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
from glob import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms


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
    r"""Returns cosine similarity between x1 and x2, computed along dim.

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


def list_files(path, ext='jpg'):
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result


def load_image(path, imsize=256, volatile=True, use_cuda=False):
    im = Image.open(path)
    im = im.convert('RGB')

    loader = transforms.Compose([
        transforms.Scale(imsize),
        transforms.ToTensor()])

    im = Variable(loader(im), volatile=volatile)
    im = im.unsqueeze(0)
    if use_cuda:
        im = im.cuda()
    return im


def data_generator(imsize=256, use_cuda=False):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'

    photo_paths = list_files(photo_dir, ext='jpg')
    sketch_paths = list_files(sketch_dir, ext='png')

    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]

        photo_filename = sketch_filename.split('-')[0] + '.jpg'
        photo_path = os.path.join(photo_dir, sketch_folder, photo_filename)

        photo = load_image(photo_path, imsize=imsize, use_cuda=use_cuda)
        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)

        yield (photo, sketch)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_name', type=str, default='conv_4_2')
    parser.add_argument('--distance', type=str, default='euclidean')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--outdir', type=str, default='./outputs')
    args = parser.parse_args()

    print('-------------------------')
    print('Layer Name: {}'.format(args.layer_name))
    print('Distance: {}'.format(args.distance))
    print('Batch Size: {}'.format(args.batch))
    print('-------------------------')
    print('')

    use_cuda = torch.cuda.is_available()
    generator = data_generator(use_cuda=use_cuda)
    layer_test = SingleLayerLossTest(args.layer_name, distance=args.distance, 
                                     use_cuda=use_cuda)

    b = 0  # number of batches
    n = 0  # number of examples
    quit = False
    loss_list = []

    if generator:
        while True:
            photo_batch = Variable(torch.zeros(args.batch, 3, 256, 256))
            sketch_batch = Variable(torch.zeros(args.batch, 3, 256, 256))
  
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

    loss_list = np.array(loss_list)
    filename = 'loss_{name}_{distance}.npy'.format(name=args.layer_name, distance=args.distance)
    np.save(os.path.join(args.outdir, filename), loss_list)
