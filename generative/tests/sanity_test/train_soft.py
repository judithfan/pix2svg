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
from sklearn.metrics import accuracy_score

from model import SketchNetSoft
from dataset import SketchPlusPhotoGroup


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
    model = SketchNetSoft()
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='./trained_models/', 
                        help='where to save model [default: ./trained_models/]')
    parser.add_argument('--batch-size', type=int, default=16, help='number of examples in a mini-batch [default: 16]')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate [default: 3e-4]')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs [default: 500]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        SketchPlusPhotoGroup(layer='fc6', soft_labels=True),
        batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        SketchPlusPhotoGroup(layer='fc6', soft_labels=True),
        batch_size=args.batch_size, shuffle=False)

    model = SketchNetSoft()
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (photo, sketch, label) in enumerate(train_loader):
            photo = Variable(photo)
            sketch = Variable(sketch)
            label = Variable(label)
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()

            photo = photo.view(batch_size * 4, 4096)
            sketch = sketch.view(batch_size * 4, 4096)
            # photo = photo.view(batch_size * 4, 512, 28, 28)
            # sketch = sketch.view(batch_size * 4, 512, 28, 28)
            label = label.view(batch_size * 4)
 
            optimizer.zero_grad()
            pred = model(photo, sketch)
            loss = F.mse_loss(pred, label, size_average=True)
            loss_meter.update(loss.data[0], batch_size)

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset), 
                    100. * batch_idx / len(train_loader), loss_meter.avg))
        
        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))


    def test():
        model.eval()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(test_loader))

        for batch_idx, (photo, sketch, label) in enumerate(test_loader):
            photo = Variable(photo, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()

            photo = photo.view(batch_size * 4, 4096)
            sketch = sketch.view(batch_size * 4, 4096)
            # photo = photo.view(batch_size * 4, 512, 28, 28)
            # sketch = sketch.view(batch_size * 4, 512, 28, 28)
            label = label.view(batch_size * 4)
            
            pred = model(photo, sketch)
            loss = F.mse_loss(pred, label, size_average=True)
            loss_meter.update(loss.data[0], batch_size)
            pbar.update()

        pbar.close()
        print('====> Test Loss: {:.4f}'.format(loss_meter.avg))
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
        }, is_best, folder=args.out_dir)
