"""Trying to non-linear or nonlinear transformation from photo 
embeddings to sketch embeddings prior to the standard distribution test.

- MLP
- Linear + Relu
- Affine (Ax + b)
- Similarity (Rotation + Translation + Dilation)
- Rigid Body (Rotation + Translation)
- Rotation
- Translation

See rotationtest.py for rest of models
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
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
        self.in_dim = in_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class NonLinearNet(nn.Module):
    def __init__(self, in_dim):
        super(NonLinearNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.in_dim = in_dim

    def forward(self, x):
        return F.tanh(self.fc1(x))


class AffineNet(nn.Module):
    def __init__(self, in_dim):
        super(AffineNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.in_dim = in_dim

    def forward(self, x):
        return self.fc1(x)


class TranslationNet(nn.Module):
    def __init__(self, in_dim):
        super(TranslationNet, self).__init__()
        self.params = Parameter(torch.normal(torch.zeros(in_dim), 1))
        self.in_dim = in_dim

    def translation_matrix(self):
        T = Variable(torch.diag(torch.ones(self.in_dim + 1)).view(-1))
        V = add_bias_1d(self.params)
        if V.is_cuda:
            T = T.cuda()
        T = torch.cat((T[:(-self.in_dim - 1)], V))
        return T.view(self.in_dim + 1, self.in_dim + 1)

    def forward(self, x):
        x = add_bias_2d(x)
        x = torch.mm(x, self.translation_matrix())
        return x[:, :-1]


class SimilarityNet(nn.Module):
    def __init__(self, in_dim):
        super(SimilarityNet, self).__init__()
        self.t_params = Parameter(torch.normal(torch.zeros(in_dim), 1))
        self.d_params = Parameter(torch.normal(torch.zeros(1), 1))
        r_params = torch.normal(torch.zeros((in_dim + 1)**2))
        self.r_params = Parameter(r_params.view(in_dim + 1, in_dim + 1))  # rotation params
        self.in_dim = in_dim

    def translation_matrix(self):
        T = Variable(torch.diag(torch.ones(self.in_dim + 1)).view(-1))
        V = add_bias_1d(self.params)
        if V.is_cuda:
            T = T.cuda()
        T = torch.cat((T[:(-self.in_dim - 1)], V))
        return T.view(self.in_dim + 1, self.in_dim + 1)

    def dilation_matrix(self):
        D = torch.diag(torch.ones(self.in_dim + 1))
        D = D * self.d_params.expand_as(D)
        return D

    def rotation_matrix(self):
        R = self.r_params
        return R

    def forward(self, x):
        x = add_bias_2d(x)
        x = torch.mm(self.translation_matrix(), x)
        x = torch.mm(self.dilation_matrix(), x)
        x = torch.mm(self.rotation_matrix(), x)
        return x[:, :-1]

    def constraint(self):
        RtR = torch.mm(torch.t(self.r_params), self.r_params)
        det = Cholesky.apply(RtR).diag().prod()
        return det - 1


class RigidBodyNet(nn.Module):
    def __init__(self, in_dim):
        super(RigidBodyNet, self).__init__() 
        self.t_params = Parameter(torch.normal(torch.zeros(in_dim), 1))  # translation params
        r_params = torch.normal(torch.zeros((in_dim + 1)**2))
        self.r_params = Parameter(r_params.view(in_dim + 1, in_dim + 1))  # rotation params
        self.in_dim = in_dim

    def translation_matrix(self):
        T = Variable(torch.diag(torch.ones(self.in_dim + 1)).view(-1))
        V = add_bias_1d(self.params)
        if V.is_cuda:
            T = T.cuda()
        T = torch.cat((T[:(-self.in_dim - 1)], V))
        return T.view(self.in_dim + 1, self.in_dim + 1)

    def rotation_matrix(self):
        R = self.r_params
        return R

    def forward(self, x):
        x = add_bias_2d(x)
        x = torch.mm(self.translation_matrix(), x)
        x = torch.mm(self.rotation_matrix(), x)
        return x[:, :-1]

    def constraint(self):
        RtR = torch.mm(torch.t(self.r_params), self.r_params)
        det = Cholesky.apply(RtR).diag().prod()
        return det - 1


class RotationNet(nn.Module):
    """Note: This isn't as strict as a real rotation matrix since 
    I don't know how to optimize such that a matrix is orthogonal. 
    Instead, I will optimize such that the determinant is 1. This 
    should be a rotation that preserves volumn but does not guarantee
    that parallel lines remain parallel.
    """
    def __init__(self, in_dim):
        super(RotationNet, self).__init__()
        params = torch.normal(torch.zeros((in_dim + 1)**2))
        self.params = Parameter(params.view(in_dim + 1, in_dim + 1))  # rotation params
        self.in_dim = in_dim

    def forward(self, x):
        x = add_bias_2d(x)
        x = torch.mm(x, self.params)
        return x[:, :-1]

    def constraint(self):
        RtR = torch.mm(torch.t(self.params), self.params)
        det = Cholesky.apply(RtR).diag().prod()
        return det - 1


def add_bias_1d(x):
    bias = Variable(torch.Tensor([1.]))
    if x.is_cuda:
        bias = bias.cuda()
    return torch.cat((x, bias))


def add_bias_2d(x, use_cuda=False):
    bias = Variable(torch.ones(x.size(0)).unsqueeze(1))
    if x.is_cuda:
        bias = bias.cuda()
    return torch.cat((x, bias), 1)


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


class Cholesky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l

    @staticmethod
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        # Gradient is l^{-H} @ ((l^{H} @ grad) * (tril(ones)-1/2*eye)) @ l^{-1}
        # TODO: ideally, this should use some form of solve triangular instead of inverse...
        linv =  l.inverse()
        
        inner = torch.tril(torch.mm(l.t(),grad_output))*torch.tril(1.0-Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        # could re-symmetrise 
        #s = (s+s.t())/2.0
        return s


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


def save_checkpoint(state, is_best, outdir, filename='checkpoint'):
    checkpoint_path = os.path.join(outdir, '{}.pth.tar'.format(filename))
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(outdir, '{}.best.pth.tar'.format(filename)))


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
    if checkpoint['net'] == 'rotation':
        model = RotationNet(checkpoint['n_dims'])
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


def cnn_predict(x, cnn):
    x = cnn.features(x)
    x = x.view(x.size(0), -1)

    classifier = list(cnn.classifier)[:4]
    for i in range(len(classifier)):
        x = classifier[i](x)
    
    return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('net', type=str, help='translation|rotation|rigidbody|similarity|affine|nonlinear|mlp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--rotation_lambda', type=float, default=100.0)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.net in ['translation', 'rotation', 'rigidbody', 'similarity', 'affine', 'nonlinear', 'mlp']

    def reset_generators():
        train_generator = train_test_generator(imsize=224, train=True, use_cuda=args.cuda)
        test_generator = train_test_generator(imsize=224, train=False, use_cuda=args.cuda)        
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    n_train, n_test = train_test_size() 

    cnn = models.vgg19()
    cnn.eval()
    has_constraint = False

    if args.net == 'translation':
        model = TranslationNet(4096)
    elif args.net == 'rotation':
        model = RotationNet(4096)
        has_constraint = True
    elif args.net == 'rigidbody':
        model = RigidBodyNet(4096)
        has_constraint = True
    elif args.net == 'similarity':
        model = SimilarityNet(4096)
        has_constraint = True
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
        constraints = AverageMeter()

        model.train()

        batch_idx = 0
        quit = False

        while True:
            photo = Variable(torch.zeros(args.batch_size, 3, 224, 224))
            sketch = Variable(torch.zeros(args.batch_size, 3, 224, 224))
  
            if args.cuda:
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
            photo_emb, sketch_emb = cnn_predict(photo, cnn), cnn_predict(sketch, cnn)
            photo_emb = model(photo_emb)
            optimizer.zero_grad()
            loss = torch.norm(photo_emb - sketch_emb, p=2)

            if has_constraint:
                constraint = args.rotation_lambda * model.constraint()
                constraints.update(constraint.data[0], b)
                loss = loss + constraint

            losses.update(loss.data[0], b)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Distance: {:.6f}\tAverage Constraint: {:.6f}'.format(
                    epoch, batch_idx * args.batch_size + (b + 1), n_train, 
                    100 * (batch_idx * args.batch_size + (b + 1)) / n_train, losses.avg, constraints.avg))

            batch_idx += 1

            if quit: 
                break

        return losses.avg


    def test(epoch):
        losses = AverageMeter()
        constraints = AverageMeter()
        model.eval()
        quit = False

        while True:
            photo = Variable(torch.zeros(args.batch_size, 3, 224, 224))
            sketch = Variable(torch.zeros(args.batch_size, 3, 224, 224))
  
            if args.cuda:
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
            photo_emb, sketch_emb = cnn_predict(photo, cnn), cnn_predict(sketch, cnn)
            photo_emb = model(photo_emb)
            
            loss = torch.norm(photo_emb - sketch_emb, p=2)
            if has_constraint:
                constraint = args.rotation_lambda * model.constraint()
                constraints.update(constraint.data[0], b)
                loss = loss + constraint

            losses.update(loss.data[0], b)

            if quit: 
                break

        print('Test Epoch: {}\tAverage Distance: {:.6f}\tAverage Constraint: {:.6f}'.format(
            epoch, losses.avg, constraints.avg))
        return losses.avg

    best_loss = sys.maxint

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        train_generator, test_generator = reset_generators()

        is_best = test_loss <= best_loss
        best_loss = max(test_loss, best_loss)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'euclidean_distance': best_loss,
            'optimizer' : optimizer.state_dict(),
            'net': args.net,
        }

        save_checkpoint(checkpoint, is_best, args.outdir, 
                        filename='model.{}'.format(args.net))
