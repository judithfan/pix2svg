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
from train_sketch import AverageMeter
from dataset_sketch import SketchOnlyDataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='where to find saved file')
    parser.add_argument('--split', type=str, default='test', help='train|val|test')
    parser.add_argument('--batch-size', type=int, default=10, 
                        help='number of examples in a mini-batch [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    test_dataset = SketchOnlyDataset(layer='conv42', split=args.split, soft_labels=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = load_checkpoint(args.model_path)
    model.eval()
    if args.cuda:
        model.cuda()

    def test():
        model.eval()
        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        pbar = tqdm(total=len(test_loader))
        for batch_idx, (sketch, label) in enumerate(test_loader):
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)
            batch_size = len(sketch)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            pred_logits = model(sketch)
            pred = F.softplus(pred_logits)
            pred = pred / torch.sum(pred, dim=1, keepdim=True)
            loss = F.mse_loss(pred, label.float())
            loss_meter.update(loss.data[0], batch_size)

            label_np = label.cpu().data.numpy()
            pred_np = pred.cpu().data.numpy()
            mse = mean_squared_error(label_np, pred_np)
            mse_meter.update(mse, batch_size)
            pbar.update()
        pbar.close()
        return loss_meter.avg, mse_meter.avg

    test_loss, test_metric = test()
    print('====> {} Loss: {:.4f}\t{} MSE: {:.2f}'.format(
        args.split.capitalize(), test_loss, args.split.capitalize(), test_metric))
