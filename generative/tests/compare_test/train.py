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
from sklearn.metrics import accuracy_score, mean_squared_error

from model import Classifier
from dataset import VisualCommunicationDataset


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
    model = Classifier(checkpoint['distance'])
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
    parser.add_argument('--alpha', type=float, default=1.,
                        help='sample from same category with this probability [default: 1.]')
    parser.add_argument('--add-class-loss', action='store_true', default=False,
                        help='add class loss [default: False]')
    parser.add_argument('--soft-labels', action='store_true', default=False,
                        help='use soft or hard labels [default: False]')
    parser.add_argument('--out-dir', type=str, default='./trained_models', 
                        help='where to save checkpoints [./trained_models]')
    parser.add_argument('--distance', type=str, default='cosine', help='cosine|euclidean')
    parser.add_argument('--batch-size', type=int, default=10, 
                        help='number of examples in a mini-batch [default: 10]')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate [default: 1e-3]')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
   
    train_dataset = VisualCommunicationDataset(layer='conv42', split='train', soft_labels=args.soft_labels, alpha=args.alpha)
    val_dataset = VisualCommunicationDataset(layer='conv42', split='val', soft_labels=args.soft_labels, alpha=args.alpha)
    test_dataset = VisualCommunicationDataset(layer='conv42', split='test', soft_labels=args.soft_labels, alpha=args.alpha)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = Classifier(args.distance)
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()
        if args.add_class_loss:
            photo_meter = AverageMeter()
            sketch_meter = AverageMeter()

        for batch_idx, (photo, sketch, label, photo_class, sketch_class) in enumerate(train_loader):
            photo = Variable(photo)
            sketch = Variable(sketch)
            label = Variable(label)
            if args.add_class_loss:
                photo_class = Variable(photo_class)
                sketch_class = Variable(sketch_class)
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()
                if args.add_class_loss:
                    photo_class = photo_class.cuda()
                    sketch_class = sketch_class.cuda()

            photo = photo.view(batch_size * 4, 512, 28, 28)
            sketch = sketch.view(batch_size * 4, 512, 28, 28)
            label = label.view(batch_size * 4, 1)
            if args.add_class_loss:
                photo_class = photo_class.view(batch_size * 4)
                sketch_class = sketch_class.view(batch_size * 4)

            optimizer.zero_grad()
            pred, photo_pred, sketch_pred = model(photo, sketch)
            if args.add_class_loss:
                loss = (F.binary_cross_entropy(pred, label) + 
                        F.cross_entropy(photo_pred, photo_class) + 
                        F.cross_entropy(sketch_pred, sketch_class))
            else:
                loss = F.binary_cross_entropy(pred, label)
            loss_meter.update(loss.data[0], batch_size)
            
            if args.soft_labels:
                label_np = label.cpu().data.numpy()
                pred_np = pred.cpu().data.numpy()
                mse = mean_squared_error(label_np, pred_np)
                metric_meter.update(mse, batch_size)
            else:
                label_np = np.round(label.cpu().data.numpy(), 0)
                pred_np = np.round(pred.cpu().data.numpy(), 0)
                acc = accuracy_score(label_np, pred_np)
                metric_meter.update(acc, batch_size)

            if args.add_class_loss:
                photo_class_np = photo_class.cpu().data.numpy()
                photo_pred_np = photo_pred.cpu().data.numpy()
                photo_pred_np = np.argmax(photo_pred_np, axis=1)
                acc = np.sum(photo_pred_np == photo_class_np) / float(4. * batch_size)
                photo_meter.update(acc, batch_size)

                sketch_class_np = sketch_class.cpu().data.numpy()
                sketch_pred_np = sketch_pred.cpu().data.numpy()
                sketch_pred_np = np.argmax(sketch_pred_np, axis=1)
                acc = np.sum(sketch_pred_np == sketch_class_np) / float(4. * batch_size)
                sketch_meter.update(acc, batch_size)

            loss.backward()
            optimizer.step()
            
            if args.add_class_loss:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}: {:6f}\tPhoto-Acc: {:.2f}\tSketch-Acc: {:.2f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),  100. * batch_idx / len(train_loader), 
                    loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy' , metric_meter.avg, photo_meter.avg, sketch_meter.avg))
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}: {:6f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),  100. * batch_idx / len(train_loader),
                    loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy' , metric_meter.avg))
        
        if args.add_class_loss:
            print('====> Epoch: {}\tLoss: {:.4f}\t{}: {:.6f}\tPhoto-Acc: {:.2f}\tSketch-Acc: {:.2f}'.format(
                epoch, loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy', metric_meter.avg,
                photo_meter.avg, sketch_meter.avg))
        else:
            print('====> Epoch: {}\tLoss: {:.4f}\t{}: {:.6f}'.format(
                epoch, loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy', metric_meter.avg))
        return loss_meter.avg, metric_meter.avg

    def validate():
        model.eval()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()
        if args.add_class_loss:
            photo_meter = AverageMeter()
            sketch_meter = AverageMeter()
        pbar = tqdm(total=len(val_loader))

        for batch_idx, (photo, sketch, label, photo_class, sketch_class) in enumerate(val_loader):
            photo = Variable(photo, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False).float()
            if args.add_class_loss:
                photo_class = Variable(photo_class, requires_grad=False)
                sketch_class = Variable(sketch_class, requires_grad=False)
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()
                if args.add_class_loss:
                    photo_class = photo_class.cuda()
                    sketch_class = sketch_class.cuda()

            photo = photo.view(batch_size * 4, 512, 28, 28)
            sketch = sketch.view(batch_size * 4, 512, 28, 28)
            label = label.view(batch_size * 4, 1)
            if args.add_class_loss:
                photo_class = photo_class.view(batch_size * 4)
                sketch_class = sketch_class.view(batch_size * 4)

            pred, photo_pred, sketch_pred = model(photo, sketch)
            if args.add_class_loss:
                loss = (F.binary_cross_entropy(pred, label) +
                        F.cross_entropy(photo_pred, photo_class) + 
                        F.cross_entropy(sketch_pred, sketch_class))
            else:
                loss = F.binary_cross_entropy(pred, label)
            loss_meter.update(loss.data[0], batch_size)

            if args.soft_labels:
                label_np = label.cpu().data.numpy()
                pred_np = pred.cpu().data.numpy()
                mse = mean_squared_error(label_np, pred_np)
                metric_meter.update(mse, batch_size)
            else:
                label_np = np.round(label.cpu().data.numpy(), 0)
                pred_np = np.round(pred.cpu().data.numpy(), 0)
                acc = accuracy_score(label_np, pred_np)
                metric_meter.update(acc, batch_size)

            if args.add_class_loss:
                photo_class_np = photo_class.cpu().data.numpy()
                photo_pred_np = photo_pred.cpu().data.numpy()
                photo_pred_np = np.argmax(photo_pred_np, axis=1)
                acc = np.sum(photo_pred_np == photo_class_np) / float(4. * batch_size)
                photo_meter.update(acc, batch_size)

                sketch_class_np = sketch_class.cpu().data.numpy()
                sketch_pred_np = sketch_pred.cpu().data.numpy()
                sketch_pred_np = np.argmax(sketch_pred_np, axis=1)
                acc = np.sum(sketch_pred_np == sketch_class_np) / float(4. * batch_size)
                sketch_meter.update(acc, batch_size) 

            pbar.update()
        pbar.close()
        if args.add_class_loss:
            print('====> Val Loss: {:.4f}\tVal {}: {:.6f}\tPhoto-Acc: {:.2f}\tSketch-Acc: {:.2f}'.format(
                loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy', metric_meter.avg,
                photo_meter.avg, sketch_meter.avg))
        else:
            print('====> Val Loss: {:.4f}\tVal {}: {:.6f}'.format(
                loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy', metric_meter.avg))
        return loss_meter.avg, metric_meter.avg

    def test():
        model.eval()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()
        if args.add_class_loss:
            photo_meter = AverageMeter()
            sketch_meter = AverageMeter()
        pbar = tqdm(total=len(test_loader))

        for batch_idx, (photo, sketch, label, photo_class, sketch_class) in enumerate(test_loader):
            photo = Variable(photo, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False).float()
            if args.add_class_loss:
                photo_class = Variable(photo_class, requires_grad=False)
                sketch_class = Variable(sketch_class, requires_grad=False)
            batch_size = len(photo)

            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()
                if args.add_class_loss:
                    photo_class = photo_class.cuda()
                    sketch_class = sketch_class.cuda()

            photo = photo.view(batch_size * 4, 512, 28, 28)
            sketch = sketch.view(batch_size * 4, 512, 28, 28)
            label = label.view(batch_size * 4, 1)
            if args.add_class_loss:
                photo_class = photo_class.view(batch_size * 4)
                sketch_class = sketch_class.view(batch_size * 4)

            pred, photo_pred, sketch_pred = model(photo, sketch)
            if args.add_class_loss:
                loss = (F.binary_cross_entropy(pred, label) +
                        F.cross_entropy(photo_pred, photo_class) +
                        F.cross_entropy(sketch_pred, sketch_class))
            else:
                loss = F.binary_cross_entropy(pred, label)
            loss_meter.update(loss.data[0], batch_size)

            if args.soft_labels:
                label_np = label.cpu().data.numpy()
                pred_np = pred.cpu().data.numpy()
                mse = mean_squared_error(label_np, pred_np)
                metric_meter.update(mse, batch_size)
            else:
                label_np = np.round(label.cpu().data.numpy(), 0)
                pred_np = np.round(pred.cpu().data.numpy(), 0)
                acc = accuracy_score(label_np, pred_np)
                metric_meter.update(acc, batch_size)
           
            if args.add_class_loss:
                photo_class_np = photo_class.cpu().data.numpy()
                photo_pred_np = photo_pred.cpu().data.numpy()
                photo_pred_np = np.argmax(photo_pred_np, axis=1)
                acc = np.sum(photo_pred_np == photo_class_np) / float(4. * batch_size)
                photo_meter.update(acc, batch_size)

                sketch_class_np = sketch_class.cpu().data.numpy()
                sketch_pred_np = sketch_pred.cpu().data.numpy()
                sketch_pred_np = np.argmax(sketch_pred_np, axis=1)
                acc = np.sum(sketch_pred_np == sketch_class_np) / float(4. * batch_size)
                sketch_meter.update(acc, batch_size)

            pbar.update()
        pbar.close()
        if args.add_class_loss:
            print('====> Test Loss: {:.4f}\tTest {}: {:.6f}\tPhoto-Acc: {:.2f}\tSketch-Acc: {:.2f}'.format(
                loss_meter.avg, 'MSE' if args.soft_labels else 'Accuracy', metric_meter.avg,
                photo_meter.avg, sketch_meter.avg))
        else:
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
            'distance': args.distance,
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
