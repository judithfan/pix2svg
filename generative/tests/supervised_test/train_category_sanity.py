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

from model import SketchNetCATEGORY
from train import save_checkpoint
from train import AverageMeter
from train import cross_entropy

sys.path.append('../conv_test')
from datasets import ContextFreePreloadedGenerator as Generator
from model_old import ConvEmbedNet

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
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate [default: 1e-3]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    data_dir = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % args.layer

    def reset_generators():
        train_generator = Generator(train=True, batch_size=args.batch_size, use_cuda=args.cuda,
                                    global_negatives=False, balance_crops=False,
                                    data_dir=data_dir)
        test_generator = Generator(train=False, batch_size=args.batch_size, use_cuda=args.cuda,
                                   global_negatives=False, data_dir=data_dir)
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    train_examples = train_generator.make_generator()
    test_examples = test_generator.make_generator() 

    # model = SketchNetCATEGORY(layer=args.layer) 
    model = ConvEmbedNet()
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        batch_idx = 0

        while True:
            try:
                photo, sketch, label, _, _ = train_examples.next()
                batch_idx += 1
            except StopIteration:
                break

            batch_size = len(photo)
            optimizer.zero_grad()
            # same_pred, cat_pred = model(photo, sketch)
            same_pred = model(photo, sketch) 

            loss = F.binary_cross_entropy(same_pred, label)
            loss_meter.update(loss.data[0], batch_size)

            label_np = label.cpu().data.numpy()
            same_pred_np = np.round(same_pred.cpu().data.numpy(), 0)
            acc = accuracy_score(label_np, same_pred_np)
            acc_meter.update(acc, batch_size)

            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                  epoch, batch_idx * args.batch_size, train_generator.size,
                  (100. * batch_idx * args.batch_size) / train_generator.size,
                  loss_meter.avg, acc_meter.avg))       
 
        print('====> Epoch: {}\tLoss: {:.4f}\tAcc: {:.2f}'.format(epoch, loss_meter.avg, acc_meter.avg))

    def test():
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        batch_idx = 0

        while True:
            try:
                photo, sketch, label, _, _ = test_examples.next()
                batch_idx += 1
            except StopIteration:
                break
            
            batch_size = len(photo)
            # same_pred, cat_pred = model(photo, sketch)            
            same_pred = model(photo, sketch)
            loss = F.binary_cross_entropy(same_pred, label)
            loss_meter.update(loss.data[0], len(photo))

            label_np = label.cpu().data.numpy()
            same_pred_np = np.round(same_pred.cpu().data.numpy(), 0)
            acc = accuracy_score(label_np, same_pred_np)
            acc_meter.update(acc, batch_size)

        print('Test Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}'.format(
              epoch, loss_meter.avg, acc_meter.avg))
        return loss_meter.avg
    
    best_loss = sys.maxint
    for epoch in xrange(1, args.epochs + 1):
        train(epoch)
        loss = test()
        
        train_generator, test_generator = reset_generators()
        train_examples = train_generator.make_generator()
        test_examples = test_generator.make_generator()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
            'layer': args.layer,
        }, is_best, folder=args.out_dir)

