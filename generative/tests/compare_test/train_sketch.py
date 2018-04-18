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

from model import LabelPredictor
from dataset import SketchOnlyDataset


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
    model = LabelPredictor()
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
    parser.add_argument('--out-dir', type=str, default='./trained_models', 
                        help='where to save checkpoints [./trained_models]')
    parser.add_argument('--soft-labels', action='store_true', default=False,
                        help='use soft or hard labels [default: False]')
    parser.add_argument('--batch-size', type=int, default=10, 
                        help='number of examples in a mini-batch [default: 10]')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate [default: 1e-3]')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
   
    train_dataset = SketchOnlyDataset(layer='conv42', split='train', soft_labels=args.soft_labels)
    val_dataset = SketchOnlyDataset(layer='conv42', split='val', soft_labels=args.soft_labels)
    test_dataset = SketchOnlyDataset(layer='conv42', split='test', soft_labels=args.soft_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = LabelPredictor()
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()

        for batch_idx, (sketch, label) in enumerate(train_loader):
            sketch = Variable(sketch)
            label = Variable(label)
            batch_size = len(sketch)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            pred_logits = model(sketch)
            if args.soft_labels:
                loss = F.mse_loss(F.softplus(pred_logits), label)
            else:
                loss = F.cross_entropy(pred_logits, label)
            loss_meter.update(loss.data[0], batch_size)
            
            if args.soft_labels:
                label_np = label.cpu().data.numpy()
                pred_np = pred.cpu().data.numpy()
                mse = mean_squared_error(label_np, pred_np)
                metric_meter.update(mse, batch_size)
            else:
                pred = pred_logits.data.max(1, keepdim=True)[1]
                correct = pred.eq(label.data.view_as(pred)).long().cpu().sum()
                metric_meter.update(correct / float(batch_size), batch_size)

            loss.backward()
            optimizer.step()
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}: {:6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),  100. * batch_idx / len(train_loader),
                loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy' , metric_meter.avg))
        
        print('====> Epoch: {}\tLoss: {:.4f}\t{}: {:.6f}'.format(
            epoch, loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy', metric_meter.avg))
        
        return loss_meter.avg, metric_meter.avg

    def validate():
        model.eval()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()
        pbar = tqdm(total=len(val_loader))

        for batch_idx, (sketch, label) in enumerate(val_loader):
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False).float()
            batch_size = len(photo)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            pred_logits = model(sketch)
            if args.soft_labels:
                loss = F.mse_loss(F.softplus(pred_logits), label)
            else:
                loss = F.cross_entropy(pred_logits, label) 
            loss_meter.update(loss.data[0], batch_size)

            if args.soft_labels:
                label_np = label.cpu().data.numpy()
                pred_np = pred.cpu().data.numpy()
                mse = mean_squared_error(label_np, pred_np)
                metric_meter.update(mse, batch_size)
            else:
                pred = pred_logits.data.max(1, keepdim=True)[1]
                correct = pred.eq(label.data.view_as(pred)).long().cpu().sum()
                metric_meter.update(correct / float(batch_size), batch_size)

            pbar.update()
        pbar.close()

        print('====> Val Loss: {:.4f}\tVal {}: {:.6f}'.format(
            loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy', metric_meter.avg))
        return loss_meter.avg, metric_meter.avg

    def test():
        model.eval()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()
        pbar = tqdm(total=len(test_loader))

        for batch_idx, (sketch, label) in enumerate(test_loader):
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False).float()
            batch_size = len(photo)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            pred_logits = model(sketch)
            if args.soft_labels:
                loss = F.mse_loss(F.softplus(pred_logits), label)
            else:
                loss = F.cross_entropy(pred_logits, label)
            loss_meter.update(loss.data[0], batch_size)

            if args.soft_labels:
                label_np = label.cpu().data.numpy()
                pred_np = pred.cpu().data.numpy()
                mse = mean_squared_error(label_np, pred_np)
                metric_meter.update(mse, batch_size)
            else:
                pred = pred_logits.data.max(1, keepdim=True)[1]
                correct = pred.eq(label.data.view_as(pred)).long().cpu().sum()
                metric_meter.update(correct / float(batch_size), batch_size)          
 
            pbar.update()
        pbar.close()

        print('====> Test Loss: {:.4f}\tTest {}: {:.6f}'.format(
            loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy', metric_meter.avg))
        return loss_meter.avg, metric_meter.avg

    loss_db = np.zeros((args.epochs, 3))
    metric_db = np.zeros((args.epochs, 3))
    best_loss = sys.maxint
    for epoch in xrange(1, args.epochs + 1):
        train_loss, train_metric = train(epoch)
        val_loss, val_metric = validate()
        test_loss, test_metric = test()
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'train_loss': train_loss,
            'train_metric': train_metric,
            'val_loss': val_loss,
            'val_metric': val_metric,
            'test_loss': test_loss,
            'test_metric': test_metric,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=args.out_dir)
        # save stuff for plots
        loss_db[epoch - 1, 0] = train_loss
        loss_db[epoch - 1, 1] = val_loss
        loss_db[epoch - 1, 2] = test_loss
        metric_db[epoch - 1, 0] = train_metric
        metric_db[epoch - 1, 1] = val_metric
        metric_db[epoch - 1, 2] = test_metric

    # plot the training numbers
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    import seaborn as sns
    sns.set_style('whitegrid')
    plt.figure()
    plt.plot(loss_db[:, 0], '-', label='Train')
    plt.plot(loss_db[:, 1], '-', label='Val')
    plt.plot(loss_db[:, 2], '-', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'loss.png'))

    plt.figure()
    plt.plot(metric_db[:, 0], '-', label='Train')
    plt.plot(metric_db[:, 1], '-', label='Val')
    plt.plot(metric_db[:, 2], '-', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE' if args.soft_labels else 'Accuracy') 
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'metric.png'))
