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

from model import SketchNet
from dataset import SketchPlusGoodBadPhoto


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='./trained_models/', 
                        help='where to save model [default: ./trained_models/]')
    parser.add_argument('--batch-size', type=int, default=16, help='number of examples in a mini-batch [default: 16]')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate [default: 1e-5]')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs [default: 500]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        SketchPlusGoodBadPhoto(layer='fc6'),
        batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        SketchPlusGoodBadPhoto(layer='fc6'),
        batch_size=args.batch_size, shuffle=False)

    model = SketchNet()
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-3)
    
    def train(epoch):
        model.train()
        e_loss_meter = AverageMeter()
        c1_loss_meter = AverageMeter()
        c2_loss_meter = AverageMeter()
        c3_loss_meter = AverageMeter()
        c1_acc_meter = AverageMeter()
        c2_acc_meter = AverageMeter()
        c3_acc_meter = AverageMeter()

        for batch_idx, (sketch, good_photo, bad_photo, good_cat, bad_cat) in enumerate(train_loader):
            sketch = Variable(sketch)
            good_photo = Variable(good_photo)
            bad_photo = Variable(bad_photo)
            good_cat = Variable(good_cat)
            bad_cat = Variable(bad_cat)
            batch_size = len(photo)

            if args.cuda:
                sketch = sketch.cuda()
                good_photo = good_photo.cuda()
                bad_photo = bad_photo.cuda()
                good_cat = good_cat.cuda()
                bad_cat = bad_cat.cuda()
 
            optimizer.zero_grad()
            (sketch_e, good_photo_e, bad_photo_e,
             sketch_c, good_photo_c, bad_photo_c) = model(sketch, good_photo, bad_photo)
            
            e_loss = F.triplet_margin_loss(sketch_e, good_photo_e, bad_photo_e, margin=15.)
            c1_loss = F.cross_entropy(sketch_c, good_cat)
            c2_loss = F.cross_entropy(good_photo_c, good_cat)
            c3_loss = F.cross_entropy(bad_photo_c, bad_cat)
            loss = e_loss + c1_loss + c2_loss + c3_loss
            
            c1_acc = np.sum(np.argmax(sketch_c.cpu().data.numpy(), axis=1) == good_cat.cpu().data.numpy()) / float(batch_size)
            c2_acc = np.sum(np.argmax(good_photo_c.cpu().data.numpy(), axis=1) == good_cat.cpu().data.numpy()) / float(batch_size)
            c3_acc = np.sum(np.argmax(bad_photo_c.cpu().data.numpy(), axis=1) == bad_cat.cpu().data.numpy()) / float(batch_size)
            
            e_loss_meter.update(e_loss.data[0], batch_size)
            c1_loss_meter.update(c1_loss.data[0], batch_size)
            c2_loss_meter.update(c2_loss.data[0], batch_size)
            c3_loss_meter.update(c3_loss.data[0], batch_size)
            c1_acc_meter.update(c1_acc, batch_size)
            c2_acc_meter.update(c2_acc, batch_size)
            c3_acc_meter.update(c3_acc, batch_size)

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: ({:.6f}|{:.6f}|{:.6f}|{:.6f})\tAcc: ({:2f}|{:2f}|{:2f})'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                    e_loss_meter.avg, c1_loss_meter.avg, c2_loss_meter.avg, c3_loss_meter.avg, 
                    c1_acc_meter.avg, c2_acc_meter.avg, c3_acc_meter.avg))
        
        print('====> Epoch: {}\tLoss: ({:.4f}|{:.4f}|{:.4f}|{:.4f})\tAcc: ({:.2f}|{:.2f}|{:.2f})'.format(
            epoch, e_loss_meter.avg, c1_loss_meter.avg, c2_loss_meter.avg, c3_loss_meter.avg, 
            c1_acc_meter.avg, c2_acc_meter.avg, c3_acc_meter.avg))


    def test():
        model.eval()
        e_loss_meter = AverageMeter()
        c1_loss_meter = AverageMeter()
        c2_loss_meter = AverageMeter()
        c3_loss_meter = AverageMeter()
        c1_acc_meter = AverageMeter()
        c2_acc_meter = AverageMeter()
        c3_acc_meter = AverageMeter()

        pbar = tqdm(total=len(test_loader))
        for batch_idx, (sketch, good_photo, bad_photo, good_cat, bad_cat) in enumerate(train_loader):
            sketch = Variable(sketch, volatile=True)
            good_photo = Variable(good_photo, volatile=True)
            bad_photo = Variable(bad_photo, volatile=True)
            good_cat = Variable(good_cat, volatile=True)
            bad_cat = Variable(bad_cat, volatile=True)
            batch_size = len(photo)

            if args.cuda:
                sketch = sketch.cuda()
                good_photo = good_photo.cuda()
                bad_photo = bad_photo.cuda()
                good_cat = good_cat.cuda()
                bad_cat = bad_cat.cuda()

            (sketch_e, good_photo_e, bad_photo_e,
             sketch_c, good_photo_c, bad_photo_c) = model(sketch, good_photo, bad_photo)
            
            e_loss = F.triplet_margin_loss(sketch_e, good_photo_e, bad_photo_e, margin=15.)
            c1_loss = F.cross_entropy(sketch_c, good_cat)
            c2_loss = F.cross_entropy(good_photo_c, good_cat)
            c3_loss = F.cross_entropy(bad_photo_c, bad_cat)

            c1_acc = np.sum(np.argmax(sketch_c.cpu().data.numpy(), axis=1) == good_cat.cpu().data.numpy()) / float(batch_size)
            c2_acc = np.sum(np.argmax(good_photo_c.cpu().data.numpy(), axis=1) == good_cat.cpu().data.numpy()) / float(batch_size)
            c3_acc = np.sum(np.argmax(bad_photo_c.cpu().data.numpy(), axis=1) == bad_cat.cpu().data.numpy()) / float(batch_size)
            
            e_loss_meter.update(e_loss.data[0], batch_size)
            c1_loss_meter.update(c1_loss.data[0], batch_size)
            c2_loss_meter.update(c2_loss.data[0], batch_size)
            c3_loss_meter.update(c3_loss.data[0], batch_size)
            c1_acc_meter.update(c1_acc, batch_size)
            c2_acc_meter.update(c2_acc, batch_size)
            c3_acc_meter.update(c3_acc, batch_size)
            pbar.update()

        pbar.close()
        print('====> Test Loss: ({:.4f}|{:.4f}|{:.4f}|{:.4f})\tTest Acc: ({:.2f}|{:.2f}|{:.2f})'.format(
            e_loss_meter.avg, c1_loss_meter.avg, c2_loss_meter.avg, c3_loss_meter.avg, 
            c1_acc_meter.avg, c2_acc_meter.avg, c3_acc_meter.avg))
        return e_loss_meter.avg + c1_loss_meter.avg + c2_loss_meter.avg + c3_loss_meter.avg
    
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
