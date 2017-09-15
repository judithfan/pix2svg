"""Runs the same tests as in affinetest but includes rotation matrices which have 
to be learned separately"""


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
from affinetest import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('net', type=str, help='rigidbody|similarity')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.net in ['rigidbody', 'similarity']

    def reset_generators():
        train_generator = train_test_generator(imsize=224, train=True, use_cuda=use_cuda)
        test_generator = train_test_generator(imsize=224, train=False, use_cuda=use_cuda)        
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    rotation_train_generator, rotation_test_generator = reset_generators()
    n_data = len(list_files('/home/jefan/full_sketchy_dataset/sketches', ext='png'))

    cnn = models.vgg19()
    cnn.eval()

    if args.net == 'rigidbody':
        model = RigidBodyNet(4096)
        rotation_model = RotationNet(4096)
    elif args.net == 'similarity':
        model = SimilarityNet(4096)
        rotation_model = RotationNet(4096)

    if args.cuda:
        cnn.cuda()
        model.cuda()
        rotation_model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    rotation_optimizer = optim.Adam(rotation_model.parameters(), lr=args.lr)

    def train(epoch):
        losses = AverageMeter()
        rotation_model.eval()
        model.train()

        batch_idx = 0
        quit = False

        rotation_matrix = rotation_model.params
        model.update_rotation(rotation_matrix.data)

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
                    epoch, batch_idx * b, n_data, 100 * batch_idx * b / n_data, losses.avg))

            if quit: 
                break

        return losses.avg


    def test(epoch):
        losses = AverageMeter()
        rotation_model.eval()
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

        print('Test Epoch: {}\tAverage Distance: {:.6f}'.format(epoch, losses.avg))
        return losses.avg


    def rotation_train(epoch, weight=100.0):
        losses = AverageMeter()
        constraints = AverageMeter()
        model.eval()
        rotation_model.train()

        batch_idx = 0
        quit = False

        while True:
            photo = Variable(torch.zeros(args.batch_size, 3, 224, 224))
            sketch = Variable(torch.zeros(args.batch_size, 3, 224, 224))
  
            if use_cuda:
                photo, sketch = photo.cuda(), sketch.cuda() 

            for b in range(args.batch_size):
                try:
                    _photo, _sketch = rotation_train_generator.next()
                    photo[b] = _photo
                    sketch[b] = _sketch
                except StopIteration:
                    quit = True
                    break

            photo, sketch = photo[:b + 1], sketch[:b + 1]
            photo_emb, sketch_emb = cnn_predict(photo), cnn_predict(sketch)
            photo_emb = model(photo_emb)
            photo_emb = rotation_model(photo_emb)

            rotation_optimizer.zero_grad()
            loss = torch.norm(photo_emb - sketch_emb, p=2)
            constraint = torch.mm(photo_emb, photo_emb) - 1
            loss = loss + weight * constraint
            
            losses.update(loss.data[0], b)
            constraints.update(constraint.data[0], b)

            loss.backward()
            rotation_optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%]\tAverage Distance: {:.6f}\tAverage Constraint: {:.6f}'.format(
                    epoch, batch_idx * b, n_data, 100 * batch_idx * b / n_data, losses.avg, constraints.avg))

            if quit: 
                break

        return losses.avg


    def rotation_test(epoch, weight=100.0):
        losses = AverageMeter()
        constraints = AverageMeter()
        model.eval()
        rotation_model.eval()

        batch_idx = 0
        quit = False

        while True:
            photo = Variable(torch.zeros(args.batch_size, 3, 224, 224))
            sketch = Variable(torch.zeros(args.batch_size, 3, 224, 224))
  
            if use_cuda:
                photo, sketch = photo.cuda(), sketch.cuda() 

            for b in range(args.batch_size):
                try:
                    _photo, _sketch = rotation_test_generator.next()
                    photo[b] = _photo
                    sketch[b] = _sketch
                except StopIteration:
                    quit = True
                    break

            photo, sketch = photo[:b + 1], sketch[:b + 1]
            photo_emb, sketch_emb = cnn_predict(photo), cnn_predict(sketch)
            photo_emb = model(photo_emb)
            photo_emb = rotation_model(photo_emb)

            loss = torch.norm(photo_emb - sketch_emb, p=2)
            constraint = torch.mm(photo_emb, photo_emb) - 1
            loss = loss + weight * constraint
            
            losses.update(loss.data[0], b)
            constraints.update(constraint.data[0], b)

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

        for r_epoch in range(1, 101):
            rotation_train_loss = rotation_train(epoch)
            rotation_test_loss = rotation_test(epoch)
            rotation_train_generator, rotation_test_generator = reset_generators()

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
