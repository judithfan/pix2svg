from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score

from convmodel import EmbedNet as ConvEmbedNet
from convmodel import save_checkpoint as save_conv_checkpoint
from deepmodel import EmbedNet as FCEmbedNet
from deepmodel import save_checkpoint as save_fc_checkpoint
from referenceutils2 import (ThreeClassPreloadedGenerator, 
                             FourClassPreloadedGenerator,
                             EntityPreloadedGenerator)

from train import AverageMeter


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('generator', type=str, help='cross|intra|entity')
    parser.add_argument('--model', type=str, help='conv_4_2|fc7', default='conv_4_2')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--photo_augment', action='store_true', default=False)
    parser.add_argument('--sketch_augment', action='store_true', default=False)
    parser.add_argument('--closer', action='store_true', default=False)
    parser.add_argument('--v96', action='store_true', default=False, help='use 96 game version')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.v96 = '96' if args.v96 else ''
    assert args.model in ['conv_4_2', 'fc7']
    assert args.generator in ['cross', 'intra', 'entity']

    if args.photo_augment and args.sketch_augment:
        raise Exception('Cannot pass both photo_augment and sketch_augment')
    if args.photo_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose%s_photo_augmented_%s' % (args.v96, args.model)
    elif args.sketch_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose%s_sketch_augmented_%s' % (args.v96, args.model)
    else:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose%s_%s' % (args.v96, args.model)

    if args.model == 'conv_4_2':
        EmbedNet = ConvEmbedNet
        save_checkpoint = save_conv_checkpoint
    elif args.model == 'fc7':
        EmbedNet = FCEmbedNet
        save_checkpoint = save_fc_checkpoint

    # choose the right generator
    if args.generator == 'cross':
        Generator = ThreeClassPreloadedGenerator
    elif args.generator == 'intra':
        Generator = FourClassPreloadedGenerator
    elif args.generator == 'entity':
        Generator = EntityPreloadedGenerator

    def reset_generators():
        train_generator = Generator(train=True, batch_size=args.batch_size, use_cuda=args.cuda,
                                    data_dir=data_dir, closer_only=args.closer)
        test_generator = Generator(train=False, batch_size=args.batch_size, use_cuda=args.cuda,
                                   data_dir=data_dir, closer_only=args.closer)
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    train_examples = train_generator.make_generator()
    test_examples = test_generator.make_generator()

    model = EmbedNet()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    def train(epoch):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        model.train()
        batch_idx = 0

        while True: 
            try:
                photos, sketches, labels = train_examples.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            loss_meter.update(loss.data[0], photos.size(0)) 

            # break tape and calculate accuracy
            labels_np = labels.cpu().data.numpy()
            outputs_np = np.round(outputs.cpu().squeeze(1).data.numpy(), 0)
            acc = accuracy_score(labels_np, outputs_np)
            acc_meter.update(acc, photos.size(0))

            # compute gradients + take a step
            loss.backward()
            optimizer.step()

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
                photos, sketches, labels = test_examples.next()
                batch_idx += 1
            except StopIteration:
                break
            
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            loss_meter.update(loss.data[0], photos.size(0))

            labels_np = labels.cpu().data.numpy()
            outputs_np = np.round(outputs.cpu().squeeze(1).data.numpy(), 0)
            acc = accuracy_score(labels_np, outputs_np)
            acc_meter.update(acc, photos.size(0))

        print('Test Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}'.format(
              epoch, loss_meter.avg, acc_meter.avg))
        return acc_meter.avg


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
        }, is_best, folder=args.out_dir)   
