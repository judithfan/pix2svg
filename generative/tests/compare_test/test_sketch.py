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
from train_sketch import load_checkpoint
from train import AverageMeter

from model import LabelPredictor
from dataset import SketchOnlyDataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='where to find saved file')
    parser.add_argument('--split', type=str, default='test', help='train|val|test')
    parser.add_argument('--soft-labels', action='store_true', default=False,
                        help='use soft or hard labels [default: False]')
    parser.add_argument('--batch-size', type=int, default=10, 
                        help='number of examples in a mini-batch [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    test_dataset = SketchOnlyDataset(layer='conv42', split=args.split, soft_labels=args.split)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = load_checkpoint(args.model_path)
    model.eval()
    if args.cuda:
        model.cuda()

    def test():
        model.eval()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()
        pbar = tqdm(total=len(test_loader))
        for batch_idx, (sketch, label) in enumerate(test_loader):
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)
            batch_size = len(sketch)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            pred_logits = model(sketch)
            import pdb; pdb.set_trace()
            if args.soft_labels:
                pred = F.softplus(pred_logits)
                pred = pred / torch.sum(pred, dim=1, keepdim=True)
                loss = F.mse_loss(pred, label.float())
            else:
                loss = F.cross_entropy(pred_logits, label)
            loss_meter.update(loss.data[0], batch_size)

            if args.soft_labels:
                pred = F.softplus(pred_logits)
                pred = pred / torch.sum(pred, dim=1, keepdim=True)
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
        return loss_meter.avg, metric_meter.avg

    test_loss, test_metric = test()
    print('====> {} Loss: {:.4f}\t{} {}: {:.2f}'.format(
        args.split.capitalize(), test_loss, args.split.capitalize(), 
        'MSE' if args.soft_labels else 'Accuracy', test_metric))

