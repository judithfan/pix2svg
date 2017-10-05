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
import copy
import random
import shutil
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models
from sklearn.metrics import accuracy_score
from precompute_vgg import list_files

from generators import MultiModalTrainGenerator
from generators import MultiModalTestGenerator

from model import EmbedNet, ConvEmbedNet

EMBED_NET_TYPE = 0
CONV_EMBED_NET_TYPE = 1


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)

    if checkpoint['type'] == EMBED_NET_TYPE:
        model = EmbedNet()
    elif checkpoint['type'] == CONV_EMBED_NET_TYPE:
        model = ConvEmbedNet()
    else:
        raise Exception('Unknown model type %d.' % checkpoint['type'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_dir', type=str)
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('out_folder', type=str)
    parser.add_argument('--convolutional', action='store_true', default=False,
                        help='If True, initialize ConvEmbedNet.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--strict', action='store_true', default=False,
                        help='if True, then consider a sketch of the same class but different photo as negative.')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    def reset_generators():
        train_generator = MultiModalTrainGenerator(args.photo_emb_dir, args.sketch_emb_dir,
                                                   batch_size=args.batch_size,
                                                   strict=args.strict, use_cuda=args.cuda)
        test_generator = MultiModalTestGenerator(args.photo_emb_dir, args.sketch_emb_dir,
                                                 batch_size=args.batch_size, 
                                                 strict=args.strict, use_cuda=args.cuda)
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    train_examples = train_generator.make_generator()
    test_examples = test_generator.make_generator()

    if args.convolutional:
        model = ConvEmbedNet()
        model_type = CONV_EMBED_NET_TYPE
    else:
        model = EmbedNet()
        model_type = EMBED_NET_TYPE

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
                photos, sketches, labels, _ = train_examples.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            loss_meter.update(loss.data[0], photos.size(0)) 
            
            labels_np = labels.cpu().data.numpy()
            outputs_np = np.round(outputs.cpu().squeeze(1).data.numpy(), 0)
            acc = accuracy_score(labels_np, outputs_np)
            acc_meter.update(acc, photos.size(0))

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                      epoch, batch_idx * args.batch_size, train_generator.size,
                      (100. * batch_idx * args.batch_size) / train_generator.size,
                      loss_meter.avg, acc_meter.avg))


    def test(epoch):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        model.eval()
        batch_idx = 0
        
        while True:
            try:
                photos, sketches, labels, _ = test_examples.next()
                batch_idx += 1
            except StopIteration:
                break
            
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            
            labels_np = labels.cpu().data.numpy()
            outputs_np = np.round(outputs.cpu().squeeze(1).data.numpy(), 0)
            acc = accuracy_score(labels_np, outputs_np)
            acc_meter.update(acc, photos.size(0))
            loss_meter.update(loss.data[0], photos.size(0))

        print('Test Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}'.format(
              epoch, loss_meter.avg, acc_meter.avg))

        return acc_meter.avg


    print('')
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = test(epoch)

        train_generator, test_generator = reset_generators()
        train_examples = train_generator.make_generator()
        test_examples = test_generator.make_generator()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            'type': model_type,
            'strict': args.strict,
        }, is_best, folder=args.out_folder)
