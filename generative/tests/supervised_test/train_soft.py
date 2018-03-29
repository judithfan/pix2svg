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
from sklearn.metrics import mean_squared_error

from model import SketchNetSOFT
from dataset import SketchPlus32PhotosSOFT

from train import save_checkpoint
from train import AverageMeter
from train import cross_entropy

def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = SketchNetSOFT(checkpoint['layer'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

def cross_entropy(input, soft_targets):
    return torch.mean(torch.sum(- soft_targets * F.log_softmax(input, dim=1), dim=1))

def binary_cross_entropy(input, soft_target):
    return torch.mean(- soft_target * torch.log(input))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('layer', type=str, help='fc6|conv42')
    parser.add_argument('--out-dir', type=str, default='./trained_models/soft_fc6', 
                        help='where to save model [default: ./trained_models/soft_fc6]')
    parser.add_argument('--batch-size', type=int, default=16, help='number of examples in a mini-batch [default: 16]')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate [default: 3e-4]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(SketchPlus32PhotosSOFT(layer=args.layer), 
                                               batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(SketchPlus32PhotosSOFT(layer=args.layer), 
                                              batch_size=args.batch_size, shuffle=False)

    model = SketchNetSOFT(layer=args.layer) 
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    def train(epoch):
        model.train()
        same_loss_meter = AverageMeter()
        cat_loss_meter = AverageMeter()
        category_meter = AverageMeter()

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
        
            photo = photo.view(batch_size * 4, 512, 28, 28)
            sketch = sketch.view(batch_size * 4, 512, 28, 28)
            label = label.view(batch_size * 4)
            category = category.view(batch_size * 4)
 
            optimizer.zero_grad()
            same_pred, cat_pred = model(photo, sketch)
            same_loss = binary_cross_entropy(same_pred, label)
            cat_loss = F.cross_entropy(cat_pred, category)
            loss = same_loss + cat_loss

            same_loss_meter.update(same_loss.data[0], batch_size)
            cat_loss_meter.update(cat_loss.data[0], batch_size)

            category_np = category.cpu().data.numpy()
            cat_pred_np = np.argmax(cat_pred.cpu().data.numpy(), axis=1)
            category_acc = accuracy_score(cat_pred_np, category_np)
            category_meter.update(category_acc, batch_size)

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: ({:.6f}|{:.6f})\tCategory-Acc: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset), 
                    100. * batch_idx / len(train_loader), same_loss_meter.avg, cat_loss_meter.avg,
                    category_meter.avg))
        
        print('====> Epoch: {}\tLoss: ({:.4f}|{:.4f})\tCategory-Acc: {:.6f}'.format(
            epoch, same_loss_meter.avg, cat_loss_meter.avg, category_meter.avg))

    def test():
        model.eval()
        same_loss_meter = AverageMeter()
        cat_loss_meter = AverageMeter()
        category_meter = AverageMeter()
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

            photo = photo.view(batch_size * 4, 512, 28, 28)
            sketch = sketch.view(batch_size * 4, 512, 28, 28)
            label = label.view(batch_size * 4)
            category = category.view(batch_size * 4, 32)

            same_pred, cat_pred = model(photo, sketch)            
            same_loss = binary_cross_entropy(same_pred, label)
            cat_loss = F.cross_entropy(cat_pred, category)
            loss = same_loss + cat_loss

            same_loss_meter.update(same_loss.data[0], batch_size)
            cat_loss_meter.update(cat_loss.data[0], batch_size)

            category_np = category.cpu().data.numpy()
            cat_pred_np = np.argmax(cat_pred.cpu().data.numpy(), axis=1)
            category_acc = accuracy_score(cat_pred_np, category_np)
            category_meter.update(category_acc, batch_size)

            pbar.update()

        pbar.close()
        print('====> Test Loss: ({:.4f}|{:.4f})\tTest Category-Acc: {:.6f}'.format(
            same_loss_meter.avg, cat_loss_meter.avg, category_meter.avg))
        return same_loss_meter.avg + cat_loss_meter.avg
    
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

