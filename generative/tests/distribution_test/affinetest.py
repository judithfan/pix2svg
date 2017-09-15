"""Trying to non-linear or nonlinear transformation from photo 
embeddings to sketch embeddings prior to the standard distribution test.

- MLP
- Linear + Relu
- Affine (Ax + b)
- Similarity (Rotation + Translation + Dilation)
- Rigid Body (Rotation + Translation)
- Translation

See rotationtest.py for rest of models
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.autograd import Variable
from generators import *

import torchvision.models as models


class MLPNet(nn.Module):
    def __init__(self, in_dim):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class NonLinearNet(nn.Module):
    def __init__(self, in_dim):
        super(NonLinearNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        return F.relu(self.fc1(x))


class AffineNet(nn.Module):
    def __init__(self, in_dim):
        super(AffineNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        return self.fc1(x)


class TranslationNet(nn.Module):
    def __init__(self, in_dim):
        super(TranslationNet, self).__init__()
        self.params = Parameter(torch.normal(torch.zeros(in_dim), 1))

    def translation_matrix(self):
        T = torch.diag(torch.ones(self.in_dim + 1))
        T[:, -1] = add_bias(self.t_params)
        return T

    def forward(self, x):
        x = add_bias(x)
        return torch.mm(x, self.translation_matrix())


class SimilarityNet(nn.Module):
    def __init__(self, in_dim):
        super(SimilarityNet, self).__init__()
        self.t_params = Parameter(torch.normal(torch.zeros(in_dim), 1))
        self.d_params = Parameter(torch.normal(torch.zeros(1), 1))
        self.rotation_matrix = None
        self.in_dim = in_dim

    def translation_matrix(self):
        T = torch.diag(torch.ones(self.in_dim + 1))
        T[:, -1] = add_bias(self.t_params)
        return T

    def dilation_matrix(self):
        D = torch.diag(torch.ones(self.in_dim + 1))
        D = D * self.d_params.expand_as(D)
        return D

    def update_rotation(self, rot_mat):
        self.rotation_matrix = rot_mat

    def forward(self, x):
        x = add_bias(x)
        x = torch.mm(self.translation_matrix(), x)
        x = torch.mm(self.dilation_matrix(), x)
        x = torch.mm(self.rotation_matrix, x)
        return x


class RigidBodyNet(nn.Module):
    def __init__(self, in_dim):
        super(RigidBodyNet, self).__init__() 
        self.t_params = Parameter(torch.normal(torch.zeros(in_dim), 1))  # translation params
        self.rotation_matrix = None
        self.in_dim = in_dim

    def translation_matrix(self):
        T = torch.diag(torch.ones(self.in_dim + 1))
        T[:, -1] = add_bias(self.t_params)
        return T

    def update_rotation(self, rot_mat):
        self.rotation_matrix = rot_mat

    def forward(self, x):
        x = add_bias(x)
        x = torch.mm(self.translation_matrix(), x)
        x = torch.mm(self.rotation_matrix, x)
        return x


class RotationNet(nn.Module):
    """Note: This isn't as strict as a real rotation matrix since 
    I don't know how to optimize such that a matrix is orthogonal. 
    Instead, I will optimize such that the determinant is 1. This 
    should be a rotation that preserves volumn but does not guarantee
    that parallel lines remain parallel.
    """
    def __init__(self, in_dim):
        super(VolumnPreservingRotationNet, self).__init__()
        params = torch.normal(torch.zeros((in_dim + 1)**2), 1)
        self.params = Parameter(r_params.view(in_dim + 1, in_dim + 1))

    def forward(self, x):
        x = add_bias(x)
        return torch.mm(x, self.params)

    def constraint(self):
        RtR = torch.mm(torch.t(self.params), self.params)
        det = torch.potrf(RtR).diag().prod()
        return det - 1


def add_bias(x):
    return torch.cat((x, torch.Tensor([1.])))


def wahba_rotation(X, Y):
    """For large dimensional spaces and large number of examples, this is
    practically unusable...

    Wahba's algorithm for solving for a rotation matrix between two
    coordinate spaces. See https://en.wikipedia.org/wiki/Wahba%27s_problem.

    :param A: vectors in references space (N x D) - Torch Tensor/Variable
    :param B: vectors in body frame (N x D) - Torch Tensor/Variable
    """
    det_X = torch.potrf(X).diag().prod()
    det_Y = torch.potrf(Y).diag().prod()
    M = torch.cat([torch.Tensor([1]), torch.Tensor([1]), det_X, det_Y])
    B = torch.mm(X, torch.t(Y))
    U, S, V = torch.svd(B)
    R = torch.mm(X, torch.mm(M, torch.t(Y)))
    return R


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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(file_path, use_cuda=False):
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path, map_location=lambda storage, location: storage)
    checkpoint = load_checkpoint(args.translator_path, use_cuda=use_cuda)
    assert checkpoint in ['translation', 'rotation', 'rigidbody', 'similarity', 'affine', 'nonlinear', 'mlp']
    if checkpoint['net'] == 'translation':
        model = TranslationNet(checkpoint['n_dims'])
        model.load_state_dict(checkpoint['state_dict'])
    elif checkpoint['net'] == 'rigidbody':
        model = RigidBodyNet(checkpoint['n_dims'])
        model.load_state_dict(checkpoint['state_dict'])
    elif checkpoint['net'] == 'similarity':
        model = SimilarityNet(checkpoint['n_dims'])
        model.load_state_dict(checkpoint['state_dict'])
    elif checkpoint['net'] == 'affine':
        model = AffineNet(checkpoint['n_dims'])
        model.load_state_dict(checkpoint['state_dict'])
    elif checkpoint['net'] == 'nonlinear':
        model = NonLinearNet(checkpoint['n_dims'])
        model.load_state_dict(checkpoint['state_dict'])
    elif checkpoint['net'] == 'mlp':
        model = MLPNet(checkpoint['n_dims'])
        model.load_state_dict(checkpoint['state_dict'])
    return model


def cnn_predict(x):
    cnn = models.vgg19()
    x = cnn.features(x)
    x = x.view(x.size(0), -1)

    classifier = list(cnn.classifier)[:4]
    for i in range(len(classifier)):
        x = classifier[i](x)
    
    return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('net', type=str, help='translation|affine|nonlinear|mlp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.net in ['translation', 'affine', 'nonlinear', 'mlp']

    def reset_generators():
        train_generator = train_test_generator(imsize=224, train=True, use_cuda=use_cuda)
        test_generator = train_test_generator(imsize=224, train=False, use_cuda=use_cuda)        
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    n_data = len(list_files('/home/jefan/full_sketchy_dataset/sketches', ext='png'))

    cnn = models.vgg19()
    cnn.eval()

    if args.net == 'translation':
        model = TranslationNet(4096)
    elif args.net == 'affine':
        model = AffineNet(4096)
    elif args.net == 'nonlinear':
        model = NonLinearNet(4096)
    elif args.net == 'mlp':
        model = MLPNet(4096)

    if args.cuda:
        cnn.cuda()
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
            photo_emb, sketch_emb = cnn_predict(photo), cnn_predict(sketch)
            photo_emb = model(photo_emb)

            optimizer.zero_grad()
            loss = torch.norm(photo_emb - sketch_emb, p=2)
            losses.update(loss.data[0], b)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%]\tAverage Distance: {:.6f}'.format(
                    epoch, batch_idx * b, n_data, 100 * batch_idx * b / n_data, loss.avg))

            if quit: 
                break

        return losses.avg


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
            photo_emb, sketch_emb = cnn_predict(photo), cnn_predict(sketch)
            photo_emb = model(photo_emb)
            
            loss = torch.norm(photo_emb - sketch_emb, p=2)
            losses.update(loss.data[0], b)

            if quit: 
                break

        print('Test Epoch: {}\tAverage Distance: {:.6f}'.format(epoch, loss.avg))
        return losses.avg

    best_loss = sys.maxint

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        train_generator, test_generator = reset_generators()

        is_best = test_loss > best_loss
        best_loss = max(test_loss, best_loss)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'euclidean_distance': best_loss,
            'optimizer' : optimizer.state_dict(),
            'net': args.net,
            'in_dim': args.in_dim,
        }

        save_checkpoint(checkpoint, is_best)
