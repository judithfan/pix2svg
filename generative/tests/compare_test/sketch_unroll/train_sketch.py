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

from model_sketch import SketchOnlyPredictor
from dataset_sketch import SketchOnlyDataset


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
    model = SketchOnlyPredictor()
    model.synthetic_labels = checkpoint['synthetic_labels']
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
    parser.add_argument('--loss-scale', type=float, default=10000., help='multiplier for loss [default: 10000.]')
    parser.add_argument('--synthetic-labels', action='store_true', default=False,
                        help='whether to use a more perfect synthetic dataset [default: False]')
    parser.add_argument('--out-dir', type=str, default='./trained_models', 
                        help='where to save checkpoints [./trained_models]')
    parser.add_argument('--batch-size', type=int, default=10, 
                        help='number of examples in a mini-batch [default: 10]')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate [default: 1e-3]')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
   
    train_dataset = SketchOnlyDataset(layer='conv42', split='train', synthetic_labels=args.synthetic_labels)
    val_dataset = SketchOnlyDataset(layer='conv42', split='val', synthetic_labels=args.synthetic_labels)
    test_dataset = SketchOnlyDataset(layer='conv42', split='test', synthetic_labels=args.synthetic_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SketchOnlyPredictor()
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch_idx, (sketch, label) in enumerate(train_loader):
            sketch = Variable(sketch)
            label = Variable(label)
            batch_size = len(sketch)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            pred_logits = model(sketch)
            loss = args.loss_scale * F.cross_entropy(pred_logits, label)
            loss_meter.update(loss.data[0], batch_size)
            
            pred = pred_logits.data.max(1, keepdim=True)[1]
            correct = pred.eq(label.data.view_as(pred)).long().cpu().sum()
            accuracy = correct / float(batch_size)
            acc_meter.update(accuracy, batch_size)

            if batch_idx % 10 == 0:
                print(F.softmax(pred_logits).data.cpu().numpy()[0], label.cpu().data.numpy()[0])

            loss.backward()
            optimizer.step()
            mean_grads = torch.mean(torch.cat([param.grad.cpu().data.contiguous().view(-1)
                                               for param in model.parameters()]))
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:6f}\t|Grad|: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),  100. * batch_idx / len(train_loader),
                loss_meter.avg, acc_meter.avg, mean_grads))
        
        print('====> Epoch: {}\tLoss: {:.4f}\tAccuracy: {:.6f}'.format(
            epoch, loss_meter.avg, acc_meter.avg))
        
        return loss_meter.avg, acc_meter.avg

    def validate():
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        pbar = tqdm(total=len(val_loader))

        for batch_idx, (sketch, label) in enumerate(val_loader):
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)
            batch_size = len(sketch)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            pred_logits = model(sketch)
            loss = args.loss_scale * F.cross_entropy(pred_logits, label)
            loss_meter.update(loss.data[0], batch_size)

            pred = pred_logits.data.max(1, keepdim=True)[1]
            correct = pred.eq(label.data.view_as(pred)).long().cpu().sum()
            accuracy = correct / float(batch_size)
            acc_meter.update(accuracy, batch_size)
            pbar.update()
        pbar.close()

        print('====> Val Loss: {:.4f}\tVal Accuracy: {:.6f}'.format(
            loss_meter.avg, acc_meter.avg))
        return loss_meter.avg, acc_meter.avg

    def test():
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        pbar = tqdm(total=len(test_loader))

        for batch_idx, (sketch, label) in enumerate(test_loader):
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)
            batch_size = len(sketch)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            pred_logits = model(sketch)
            loss = args.loss_scale * F.cross_entropy(pred_logits, label)
            loss_meter.update(loss.data[0], batch_size)

            pred = pred_logits.data.max(1, keepdim=True)[1]
            correct = pred.eq(label.data.view_as(pred)).long().cpu().sum()
            accuracy = correct / float(batch_size)
            acc_meter.update(accuracy, batch_size)
            pbar.update()
        pbar.close()

        print('====> Test Loss: {:.4f}\tTest Accuracy: {:.6f}'.format(
            loss_meter.avg, acc_meter.avg))
        return loss_meter.avg, acc_meter.avg

    loss_db = np.zeros((args.epochs, 3))
    acc_db = np.zeros((args.epochs, 3))
    best_loss = sys.maxint
    for epoch in xrange(1, args.epochs + 1):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate()
        test_loss, test_acc = test()
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'synthetic_labels': args.synthetic_labels, 
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=args.out_dir)
        # save stuff for plots
        loss_db[epoch - 1, 0] = train_loss
        loss_db[epoch - 1, 1] = val_loss
        loss_db[epoch - 1, 2] = test_loss
        acc_db[epoch - 1, 0] = train_acc
        acc_db[epoch - 1, 1] = val_acc
        acc_db[epoch - 1, 2] = test_acc

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
    plt.plot(acc_db[:, 0], '-', label='Train')
    plt.plot(acc_db[:, 1], '-', label='Val')
    plt.plot(acc_db[:, 2], '-', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy') 
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'accuracy.png'))
