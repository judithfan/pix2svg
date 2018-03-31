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
from torchvision import transforms

from model import SketchNetRAW
from dataset import SketchPlus32PhotosRAW

from train import save_checkpoint
from train import AverageMeter

def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = SketchNetRAW()
    model.load_state_dict(checkpoint['state_dict'])
    return model

def binary_cross_entropy(input, soft_target):
    return torch.mean(- soft_target * torch.log(input))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='./trained_models/soft_fc6', 
                        help='where to save model [default: ./trained_models/soft_fc6]')
    parser.add_argument('--batch-size', type=int, default=16, help='number of examples in a mini-batch [default: 16]')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate [default: 3e-4]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    preprocess_data = transforms.Compose([transforms.Resize(64),
                                          transforms.CenterCrop(64),
                                          transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(SketchPlus32PhotosRAW(photo_transform=preprocess_data,
                                                                     sketch_transform=preprocess_data), 
        batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(SketchPlus32PhotosRAW(photo_transform=preprocess_data,
                                                                    sketch_transform=preprocess_data), 
        batch_size=args.batch_size, shuffle=False)

    model = SketchNetRAW() 
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (photo, sketch, label) in enumerate(train_loader):
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()

            photo = Variable(photo)
            sketch = Variable(sketch)
            label = Variable(label)
        
            photo = photo.view(batch_size * 4, 3, 64, 64)
            sketch = sketch.view(batch_size * 4, 1, 64, 64)
            label = label.view(batch_size * 4)
 
            optimizer.zero_grad()
            pred = model(photo, sketch)
            loss = binary_cross_entropy(pred, label)

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
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()

            photo = Variable(photo, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)

            photo = photo.view(batch_size * 4, 3, 64, 64)
            sketch = sketch.view(batch_size * 4, 1, 64, 64)
            label = label.view(batch_size * 4)

            pred = model(photo, sketch)            
            loss = binary_cross_entropy(pred, label)
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
