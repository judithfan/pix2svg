from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from sklearn.metrics import accuracy_score

from model import SketchNet
from dataset import SketchPlus32Photos 


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
    parser.add_argument('--out-dir', type=str, default='./trained_models', 
                        help='where to save model [default: ./trained_models]')
    parser.add_argument('--batch-size', type=int, default=32, help='number of examples in a mini-batch [default: 32]')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate [default: 3e-4]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    loader = torch.utils.data.DataLoader(SketchPlus32Photos, batch_size=args.batch_size, shuffle=True)
    model = SketchNet() 
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    def train(epoch):
        model.train()
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
            
            distances = model(photos, sketch)
            loss = F.mse_loss(distances, label)
            loss_meter.update(loss.data[0], len(photos))

            loss.backward()
            optimizer.step()

            if batch_idx & args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size,
                      len(loader.dataset), 100. * batch_idx / len(loader), loss_meter.avg))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))

    
    best_loss = sys.maxint
    for epoch in xrange(1, args.epochs + 1):
        train(epoch)
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'checkpoint.pth.tar'))

