from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from tqdm import tqdm

import torch
from torch.autograd import Variable

from model import SketchNet
from dataset import SketchPlus32Photos 
from train import AverageMeter
from train import load_checkpoint
from train import cross_entropy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./trained_models/model_best.pth.tar',
                        help='where trained models are stored.')
    parser.add_argument('--batch-size', type=int, default=64, help='number of examples in a mini-batch [default: 64]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()
    loader = torch.utils.data.DataLoader(SketchPlus32Photos(layer=model.layer), 
                                         batch_size=args.batch_size, shuffle=False) 
    
    def test():
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(loader))
        for batch_idx, (photos, sketch, label) in enumerate(loader):
            photos = Variable(photos, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)
            batch_size = len(photos)
            if args.cuda:
                photos = photos.cuda()
                sketch = sketch.cuda()
                label = label.cuda()
            log_distance = model(photos, sketch)
            loss = torch.mean(torch.sum(cross_entropy(log_distance, label), dim=1))
            loss_meter.update(loss.data[0], batch_size)
            pbar.update()
        pbar.close()
        return loss_meter.avg

    loss = test()
    print(loss)

