from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score

from model import ConvEmbedNet, FCEmbedNet
from model import save_checkpoint
from datasets import ContextFreePreloadedGenerator as Generator


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--layer', type=str, help='conv_4_2|fc7 (default: conv_4_2)', 
                        default='conv_4_2')
    parser.add_argument('--batch_size', type=int, default=25, help='(default: batch_size)')
    parser.add_argument('--lr', type=float, default=1e-3, help='(default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=20, help='(default: 20)')
    parser.add_argument('--photo_augment', action='store_true', default=False)
    parser.add_argument('--global_negatives', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.layer in ['conv_4_2', 'fc7']

    if args.photo_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose96_photo_augmented_%s' % args.layer
    else:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose96_%s' % args.layer

    EmbedNet = ConvEmbedNet if args.layer == 'conv_4_2' else FCEmbedNet

    def reset_generators():
        train_generator = Generator(train=True, batch_size=args.batch_size, use_cuda=args.cuda, 
                                    global_negatives=args.global_negatives, data_dir=data_dir)
        test_generator = Generator(train=False, batch_size=args.batch_size, use_cuda=args.cuda, 
                                   global_negatives=args.global_negatives, data_dir=data_dir)
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
                photos, sketches, labels, categories, instances = train_examples.next()
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
                photos, sketches, labels, categories, instances = test_examples.next()
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
            'layer': args.layer
        }, is_best, folder=args.out_dir)   
