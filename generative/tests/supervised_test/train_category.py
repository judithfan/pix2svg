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

from model import SketchNetCATEGORY
from dataset import SketchPlus32PhotosCATEGORY

from train import save_checkpoint
from train import AverageMeter
from train import cross_entropy

def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = SketchNetCATEGORY(checkpoint['layer'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('layer', type=str, help='fc6|conv42')
    parser.add_argument('--out-dir', type=str, default='./trained_models/category_fc6', 
                        help='where to save model [default: ./trained_models/category_fc6]')
    parser.add_argument('--batch-size', type=int, default=16, help='number of examples in a mini-batch [default: 16]')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate [default: 3e-4]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(SketchPlus32PhotosCATEGORY(layer=args.layer), 
                                               batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(SketchPlus32PhotosCATEGORY(layer=args.layer), 
                                              batch_size=args.batch_size, shuffle=False)

    model = SketchNetCATEGORY(layer=args.layer) 
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    def train(epoch):
        model.train()
        train_loss = 0
        loss_meter = AverageMeter()

        for batch_idx, (photo, sketch, label, category) in enumerate(train_loader):
            photo = Variable(photo)
            sketch = Variable(sketch)
            label = Variable(label)
            category = Variable(category)
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()
                category = category.cuda()
          
            photo = photo.view(batch_size * 4, 4096)
            sketch = sketch.view(batch_size * 4, 4096)
            label = label.view(batch_size * 4)
 
            optimizer.zero_grad()
            same_pred, cat_pred = model(photo, sketch)
            
            same_loss = F.binary_cross_entropy(same_pred, label)
            cat_loss = torch.mean(torch.sum(cross_entropy(cat_pred, category), dim=1))
            loss = same_loss + cat_loss
            # use my own x-ent to compare soft-labels against distance
            loss_meter.update(loss.data[0], len(photo))
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

        for batch_idx, (photo, sketch, label, category) in enumerate(test_loader):
            photo = Variable(photo, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)
            category = Variable(category, volatile=True)
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()
                category = category.cuda()

            photo = photo.view(batch_size * 4, 4096)
            sketch = sketch.view(batch_size * 4, 4096)
            label = label.view(batch_size * 4)

            same_pred, cat_pred = model(photo, sketch)            
            same_loss = F.binary_cross_entropy(same_pred, label)
            cat_loss = torch.mean(torch.sum(cross_entropy(cat_pred, category), dim=1))
            loss = same_loss + cat_loss

            loss_meter.update(loss.data[0], len(photo))
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

