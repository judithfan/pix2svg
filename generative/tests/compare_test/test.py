from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from train import AverageMeter
from train import load_checkpoint
from dataset import SketchPlusPhotoDataset
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='where trained models are stored.')
    # parser.add_argument('--soft-labels', action='store_true', default=False,
    #                     help='use soft or hard labels [default: False]')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='number of examples in a mini-batch [default: 64]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()
    test_loader = torch.utils.data.DataLoader(
        SketchPlusPhotoDataset(layer='fc6', split='test', soft_labels=False),
        batch_size=args.batch_size, shuffle=False)
    
    def test():
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
        return loss_meter.avg, acc_meter.avg

    test_loss, test_acc = test()
    print('====> Test Loss: {:.4f}\tTest Acc: {:.2f}'.format(test_loss, test_acc))
