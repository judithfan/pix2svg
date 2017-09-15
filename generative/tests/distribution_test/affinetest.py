"""Trying to non a linear or nonlinear transformation from photo 
embeddings to sketch embeddings prior to the standard distribution test.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from distribtest import data_generator

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.autograd import Variable
from distribtest import data_generator
from distribtest import list_files, load_image


class TranslationTransformNet(nn.Module):
    def __init__(self, in_dim):
        super(TranslationTransformNet, self).__init__()
        self.params = Parameter(torch.normal(torch.zeros(in_dim), 1))

    def translation_matrix(self):
        T = torch.diag(torch.ones(self.in_dim + 1))
        T[:, -1] = add_bias(self.t_params)
        return T

    def forward(self, x):
        x = torch.cat((x, torch.Tensor([1])))
        return torch.mm(x, self.translation_matrix())


class AffineTransformNet(nn.Module):
    def __init__(self, in_dim):
        super(AffineTransformNet, self).__init__()
        self.t_params = Parameter(torch.normal(torch.zeros(in_dim), 1))
        self.d_params = Parameter(torch.normal(torch.zeros(1), 1))
        self.in_dim = in_dim

    def translation_matrix(self):
        T = torch.diag(torch.ones(self.in_dim + 1))
        T[:, -1] = add_bias(self.t_params)
        return T

    def dilation_matrix(self):
        D = torch.diag(torch.ones(self.in_dim + 1))
        D = D * self.d_params.expand_as(D)
        return D

    def rotation_matrix(self):
        # TODO: how do set this up?
        pass

    def forward(self, x):
        x = torch.cat(x, torch.Tensor([1]))
        return torch.mm(x, self.params)


class MLPTransformNet(nn.Module):
    def __init__(self, in_dim):
        super(MLPTransformNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def add_bias(x):
    return torch.cat((x, torch.Tensor([1.])))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def data_generator(imsize=256, train=True, use_cuda=False):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'

    categories = os.listdir(sketch_dir)
    n_categories = len(categories)
    if train:
        categories = categories[:int(n_categories * 0.8)]
    else:
        categories = categories[int(n_categories * 0.8):]

    photo_paths = [path for path in list_files(photo_dir, ext='jpg') 
                   if os.path.dirname(path).split('/')[-1] in categories]
    sketch_paths = [path for path in list_files(sketch_dir, ext='png') 
                   if os.path.dirname(path).split('/')[-1] in categories]

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
    parser.add_argument('net', type=str, help='translation|affine|mlp')
    parser.add_argument('embedding_size', type=int)
    parser.add_argument('--layer_name', type=str, default='conv_4_2')
    parser.add_argument('--distance', type=str, default='euclidean')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.net in ['translation', 'affine', 'mlp']

    train_generator = data_generator(imsize=224, train=True, use_cuda=use_cuda)
    test_generator = data_generator(imsize=224, train=False, use_cuda=use_cuda)
    n_data = len(list_files('/home/jefan/full_sketchy_dataset/sketches', ext='png'))

    if args.net == 'translation':
        model = TranslationTransformNet(args.embedding_size)
    elif args.net == 'affine':
        model = AffineTransformNet(args.embedding_size)
    elif args.net == 'mlp':
        model = MLPTransformNet(args.embedding_size)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    def train(epoch):
        losses = AverageMeter()
        model.train()

        batch_idx = 0
        quit = False

        while True:
            photo = Variable(torch.zeros(args.batch_size, 3, 224, 224))
            sketch = Variable(torch.zeros(args.batch_size, 3, 224, 224))
  
            if use_cuda:
                photo, sketch = photo.cuda(), sketch.cuda() 

            for b in range(args.batch_size):
                try:
                    _photo, _sketch = train_generator.next()
                    photo[b] = _photo
                    sketch[b] = _sketch
                except StopIteration:
                    quit = True
                    break

            photo, sketch = photo[:b + 1], sketch[:b + 1]

            optimizer.zero_grad()
            photo_out = model(photo)
            sketch_out = model(sketch)
            
            loss = torch.norm(photo_out - sketch_out, p=2)
            losses.update(loss.data[0], b)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%]\tAverage Distance: {:.6f}'.format(
                    epoch, batch_idx * b, n_data, 100 * batch_idx * b / n_data, loss.avg))

            if quit: 
                break


    def test(epoch):
        losses = AverageMeter()
        model.eval()

        batch_idx = 0
        quit = False

        while True:
            photo = Variable(torch.zeros(args.batch_size, 3, 224, 224))
            sketch = Variable(torch.zeros(args.batch_size, 3, 224, 224))
  
            if use_cuda:
                photo, sketch = photo.cuda(), sketch.cuda() 

            for b in range(args.batch_size):
                try:
                    _photo, _sketch = test_generator.next()
                    photo[b] = _photo
                    sketch[b] = _sketch
                except StopIteration:
                    quit = True
                    break

            photo, sketch = photo[:b + 1], sketch[:b + 1]
            photo_out = model(photo)
            sketch_out = model(sketch)
            
            loss = torch.norm(photo_out - sketch_out, p=2)
            losses.update(loss.data[0], b)

            if quit: 
                break

        print('Test Epoch: {}\tAverage Distance: {:.6f}'.format(epoch, loss.avg))


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

