from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from sklearn.metrics import accuracy_score

from model import SketchNet
from dataset import SketchPlus32Photos 

def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))

def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = SketchNet()
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


def cross_entropy(log_input, target):
    if not (target.size(0) == log_input.size(0)):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(0), log_input.size(0)))
    loss = Variable(log_input.data.new(log_input.size()))
    K = log_input.size(1) # number of classes
    for i in xrange(K):
        loss[:, i] = target[:, i] * log_input[:, i] 
    return -loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('layer', type=str, help='fc6|conv42')
    parser.add_argument('--out-dir', type=str, default='./trained_models', 
                        help='where to save model [default: ./trained_models]')
    parser.add_argument('--batch-size', type=int, default=64, help='number of examples in a mini-batch [default: 64]')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate [default: 1e-3]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(SketchPlus32Photos(layer=args.layer), 
                                               batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(SketchPlus32Photos(layer=args.layer), 
                                              batch_size=args.batch_size, shuffle=False)

    model = SketchNet(layer=args.layer) 
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    def train(epoch):
        model.train()
        train_loss = 0
        loss_meter = AverageMeter()

        for batch_idx, (photos, sketch, label) in enumerate(train_loader):
            photos = Variable(photos)
            sketch = Variable(sketch)
            label = Variable(label, requires_grad=False)
            batch_size = len(photos)

            if args.cuda:
                photos = photos.cuda()
                sketch = sketch.cuda()
                label = label.cuda()
            
            optimizer.zero_grad()
            log_distance = model(photos, sketch)
            # use my own x-ent to compare soft-labels against distance
            loss = torch.mean(torch.sum(cross_entropy(log_distance, label), dim=1))
            loss_meter.update(loss.data[0], len(photos))
            train_loss += loss.data[0]

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size,
                      len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss_meter.avg))
        
        train_loss /= len(train_loader)
        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))

    def test():
        model.eval()
        loss_meter = AverageMeter()
        test_loss = 0
        pbar = tqdm(total=len(test_loader))
        for batch_idx, (photos, sketch, label) in enumerate(test_loader):
            photos = Variable(photos, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)
            batch_size = len(photos)
            if args.cuda:
                photos = photos.cuda()
                sketch = sketch.cuda()
                label = label.cuda()
            log_distance = model(photos, sketch)
            loss = torch.mean(torch.sum(cross_entropy(log_distance, label), dim=1))
            loss_meter.update(loss.data[0], len(photos))
            test_loss += loss.data[0]
            pbar.update()
        pbar.close()
        print('====> Test Loss: {:.4f}'.format(loss_meter.avg))
        test_loss /= len(test_loader)
        return loss_meter.avg
    
    best_loss = sys.maxint
    for epoch in xrange(1, args.epochs + 1):
        train(epoch)
        loss = test()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
            'layer': args.layer,
        }, is_best, folder=args.out_dir)

