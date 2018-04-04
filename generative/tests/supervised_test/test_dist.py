from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import SketchPlus32PhotosSOFT
from train import AverageMeter
from train_dist import load_checkpoint
from model import cosine_similarity


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./trained_models/dist_fc6/model_best.pth.tar',
                        help='where trained models are stored.')
    parser.add_argument('--batch-size', type=int, default=64, help='number of examples in a mini-batch [default: 64]')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()
    loader = torch.utils.data.DataLoader(SketchPlus32PhotosSOFT(layer=model.layer), 
                                         batch_size=args.batch_size, shuffle=False) 
    
    def test():
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(loader))
        for batch_idx, (photo, sketch, label, category) in enumerate(loader):
            photo = Variable(photo, volatile=True)
            sketch = Variable(sketch, volatile=True)
            label = Variable(label, requires_grad=False)
            batch_size = len(photo)
            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
                label = label.cuda()
            photo = photo.view(batch_size * 4, 4096)
            sketch = sketch.view(batch_size * 4, 4096)
            label = label.view(batch_size * 4)
            photo, sketch = model(photo, sketch)
            import pdb; pdb.set_trace()
            pred = F.sigmoid(cosine_similarity(photo, sketch))
            loss = F.binary_cross_entropy(pred, label)            
            loss_meter.update(loss.data[0], batch_size)
            pbar.update()
        pbar.close()
        return loss_meter.avg

    loss = test()
    print(loss)

