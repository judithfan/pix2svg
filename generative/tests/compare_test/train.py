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
from model import (ModelA, ModelB, ModelC, ModelD, ModelE, ModelF,
                   ModelG, ModelH, ModelI, ModelJ, ModelK)
from dataset import (SketchPlusPhotoDataset, ObjectSplitDataset)


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
    model_type = checkpoint['modelType']
    layer_type = checkpoint['layerType']
    if model_type == 'ModelA':
        model = ModelA(layer_type)
    elif model_type == 'ModelB':
        model = ModelB(layer_type)
    elif model_type == 'ModelC':
        model = ModelC(layer_type)
    elif model_type == 'ModelD':
        model = ModelD(layer_type)
    elif model_type == 'ModelE':
        model = ModelE(layer_type)
    elif model_type == 'ModelF':
        model = ModelF(layer_type)
    elif model_type == 'ModelG':
        model = ModelG(layer_type)
    elif model_type == 'ModelH':
        model = ModelH(layer_type)
    elif model_type == 'ModelI':
        model = ModelI(layer_type)
    elif model_type == 'ModelJ':
        model = ModelJ(layer_type)
    elif model_type == 'ModelK':
        model = ModelK(layer_type)
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
    parser.add_argument('model', type=str, help='ModelA|ModelB|ModelC|ModelD|ModelE|ModelF|ModelG|ModelH|ModelI|ModelJ|ModelK')
    # parser.add_argument('--soft-labels', action='store_true', default=False,
    #                     help='use soft or hard labels [default: False]')
    parser.add_argument('dataset', type=str, help='Object|Trial')
    parser.add_argument('layer', type=str, help='fc6|conv42')
    parser.add_argument('--out-dir', type=str, default='./trained_models', help='where to save checkpoints [./trained_models]')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='number of examples in a mini-batch [default: 64]')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate [default: 1e-5]')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs [default: 500]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.out_dir = os.path.join(args.out_dir, args.model)
    assert args.dataset in ['Object', 'Trial']
    assert args.layer in ['fc6', 'conv42']
    
    if args.dataset == 'Trial':
        train_dataset = SketchPlusPhotoDataset(layer=args.layer, split='train', soft_labels=False)
        val_dataset = SketchPlusPhotoDataset(layer=args.layer, split='val', soft_labels=False)
        test_dataset = SketchPlusPhotoDataset(layer=args.layer, split='test', soft_labels=False)
    else:
        train_dataset = ObjectSplitDataset(layer=args.layer, split='train', soft_labels=False)
        val_dataset = ObjectSplitDataset(layer=args.layer, split='val', soft_labels=False)
        test_dataset = ObjectSplitDataset(layer=args.layer, split='test', soft_labels=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'ModelA':
        model = ModelA(args.layer)
    elif args.model == 'ModelB':
        model = ModelB(args.layer)
    elif args.model == 'ModelC':
        model = ModelC(args.layer)
    elif args.model == 'ModelD':
        model = ModelD(args.layer)
    elif args.model == 'ModelE':
        model = ModelE(args.layer)
    elif args.model == 'ModelF':
        model = ModelF(args.layer)
    elif args.model == 'ModelG':
        model = ModelG(args.layer)
    elif args.model == 'ModelH':
        model = ModelH(args.layer)
    elif args.model == 'ModelI':
        model = ModelI(args.layer)
    elif args.model == 'ModelJ':
        model = ModelJ(args.layer)
    elif args.model == 'ModelK':
        model = ModelK(args.layer)

    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch_idx, (photo, sketch, label) in enumerate(train_loader):
            photo = Variable(photo)
            sketch = Variable(sketch)
            label = Variable(label).float()
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            pred = model(photo, sketch)
            loss = F.binary_cross_entropy(pred, label.unsqueeze(1))
            loss_meter.update(loss.data[0], batch_size)

            label_np = np.round(label.cpu().data.numpy(), 0)
            pred_np = np.round(pred.cpu().data.numpy(), 0).ravel()
            acc = accuracy_score(label_np, pred_np)
            acc_meter.update(acc, batch_size)

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:2f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset), 
                    100. * batch_idx / len(train_loader), loss_meter.avg, acc_meter.avg))
        
        print('====> Epoch: {}\tLoss: {:.4f}\tAcc: {:.2f}'.format(epoch, loss_meter.avg, acc_meter.avg))
	return loss_meter.avg, acc_meter.avg

    def validate():
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        pbar = tqdm(total=len(val_loader))

        for batch_idx, (photo, sketch, label) in enumerate(val_loader):
            photo = Variable(photo, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False).float()
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()

            pred = model(photo, sketch)
            loss = F.binary_cross_entropy(pred, label.unsqueeze(1))
            loss_meter.update(loss.data[0], batch_size)

            label_np = np.round(label.cpu().data.numpy(), 0)
            pred_np = np.round(pred.cpu().data.numpy(), 0).ravel()
            acc = accuracy_score(label_np, pred_np)
            acc_meter.update(acc, batch_size)
            pbar.update()

        pbar.close()
        print('====> Val Loss: {:.4f}\tVal Acc: {:.2f}'.format(loss_meter.avg, acc_meter.avg))
        return loss_meter.avg, acc_meter.avg

    def test():
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        pbar = tqdm(total=len(test_loader))

        for batch_idx, (photo, sketch, label) in enumerate(test_loader):
            photo = Variable(photo, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False).float()
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()

            pred = model(photo, sketch)
            loss = F.binary_cross_entropy(pred, label.unsqueeze(1))
            loss_meter.update(loss.data[0], batch_size)

            label_np = np.round(label.cpu().data.numpy(), 0)
            pred_np = np.round(pred.cpu().data.numpy(), 0).ravel()
            acc = accuracy_score(label_np, pred_np)
            acc_meter.update(acc, batch_size)
            pbar.update()

        pbar.close()
        print('====> Test Loss: {:.4f}\tTest Acc: {:.2f}'.format(loss_meter.avg, acc_meter.avg))
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
            'modelType': args.model,
            'datasetType': args.dataset,
            'layerType': args.layer,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=args.out_dir)
        # save stuff for plots
        loss_db[epoch - 1, 0] = train_loss
        loss_db[epoch - 1, 1] = val_loss
        loss_db[epoch - 1, 2] = test_loss
        acc_db[epoch - 1, 0] = train_acc
        acc_db[epoch - 1, 1] = val_acc
        acc_db[epoch - 1, 2] = test_acc
        # fresh pair of negative samples
        # do not reinstantiate the train_dataset b/c that changes a lot of random choices
        # train_dataset.preprocess_data()
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # no need to reload validation or testing

    # plot the training numbers
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    import seaborn as sns
    sns.set_style('whitegrid')
    plt.figure()
    plt.plot(loss_db[:, 0], '-', label='Train')
    plt.plot(loss_db[:, 1], '-', label='Val')
    plt.plot(loss_db[:, 2], '-', label='Test')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'loss.png'))

    plt.figure()
    plt.plot(acc_db[:, 0], '-', label='Train')
    plt.plot(acc_db[:, 1], '-', label='Val')
    plt.plot(acc_db[:, 2], '-', label='Test')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'accuracy.png'))
