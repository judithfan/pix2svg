"""A linear/nonlinear transformation from sketch to photo may 
not be enough -- the one-way projection may be too difficult.
We can instead project both the sketch and the photo into a 
shared embeddding space. Then we can randomly sample negatives 
from the same class and from the different class.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import copy
import random
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.autograd import Variable

import torchvision.models as models
from sklearn.metrics import accuracy_score

sys.path.append('../multimodal_test')
from multimodaltest import EmbedNet
from multimodaltest import generator_size
from multimodaltest import save_checkpoint
from multimodaltest import AverageMeter
from multimodaltest import EmbeddingGenerator


class DualSketchEmbeddingGenerator(EmbeddingGenerator):

    def gen_photo_from_sketch_filename(sketch_filename):
        return sketch_filename


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_folder', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--adaptive_size', type=int, default=1000,
                        help='size to of shared vector space for images and text [default: 1000]')
    parser.add_argument('--strict', action='store_true', default=False,
                        help='if True, then consider a sketch of the same class but different photo as negative.')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    photo_emb_dir = '/home/wumike/partial_sketch_embeddings'
    sketch_emb_dir = '/home/wumike/partial_sketch_diff_embeddings'

    def reset_generators():
        train_generator = DualSketchEmbeddingGenerator(photo_emb_dir, sketch_emb_dir, imsize=256, 
                                                       batch_size=args.batch_size, train=True, 
                                                       strict=args.strict, use_cuda=args.cuda)
        test_generator = DualSketchEmbeddingGenerator(photo_emb_dir, sketch_emb_dir, imsize=256, 
                                                      batch_size=args.batch_size, train=False, 
                                                      strict=args.strict, use_cuda=args.cuda)
        return train_generator.make_generator(), test_generator.make_generator()

    train_generator, test_generator = reset_generators()
    n_train = generator_size(sketch_emb_dir, train=True)
    n_test = generator_size(sketch_emb_dir, train=False)

    model = EmbedNet(args.adaptive_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()


    def train(epoch):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        model.train()
        batch_idx = 0
        
        while True:
            try:
                photos, sketches, labels = train_generator.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            loss_meter.update(loss.data[0], len(photos)) 
            
            if args.cuda:
                acc = accuracy_score(labels.cpu().data.numpy(),
                                     np.round(outputs.cpu().squeeze(1).data.numpy(), 0))
            else:
                acc = accuracy_score(labels.data.numpy(),
                                     np.round(outputs.squeeze(1).data.numpy(), 0))
            acc_meter.update(acc, len(photos))

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                      epoch, batch_idx * args.batch_size, n_train,
                      (100. * batch_idx * args.batch_size) / n_train,
                      loss_meter.avg, acc_meter.avg))


    def test(epoch):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        model.eval()
        batch_idx = 0
        
        while True:
            try:
                photos, sketches, labels = test_generator.next()
                batch_idx += 1
            except StopIteration:
                break
            
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            
            if args.cuda:
                acc = accuracy_score(labels.cpu().data.numpy(),
                                     np.round(outputs.cpu().squeeze(1).data.numpy(), 0))
            else:
                acc = accuracy_score(labels.data.numpy(),
                                     np.round(outputs.squeeze(1).data.numpy(), 0))
            
            acc_meter.update(acc, len(photos))
            loss_meter.update(loss.data[0], len(photos))

        print('Test Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}'.format(
              epoch, loss_meter.avg, acc_meter.avg))

        return acc_meter.avg


    print('')
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = test(epoch)

        train_generator, test_generator = reset_generators()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'adaptive_size': args.adaptive_size,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
        }, is_best, folder=args.out_folder)
