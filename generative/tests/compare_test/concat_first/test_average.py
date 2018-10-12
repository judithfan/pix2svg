from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from train import load_checkpoint, AverageMeter
from dataset import VisualDataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str,
                        help='path to trained model file.')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='number of examples in a mini-batch [default: 10]')
    parser.add_argument('--train-test-split-dir', type=str, default='./train_test_split/1',
                        help='where to load train_test_split paths [default: ./train_test_split/1]')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.checkpoint_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model = model.cuda()

    test_dataset = VisualDataset(layer=model.layer, split='test', average_labels=True, 
                                 train_test_split_dir=args.train_test_split_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    acc_meter = AverageMeter()
    pbar = tqdm(total=len(test_loader))

    for batch_idx, (sketch, label) in enumerate(test_loader):
        sketch = Variable(sketch, volatile=True)
        label = Variable(label.float(), requires_grad=False)
        batch_size = len(sketch)

        if args.cuda:
            sketch = sketch.cuda()
            label = label.cuda()

        pred_logits = []
        photo_generator = test_dataset.gen_photos()
        for photo in photo_generator():
            photo = Variable(photo)
            if args.cuda:
                photo = photo.cuda()
            photo = (photo.repeat(batch_size, 1) if model.layer == 'fc6' else
                     photo.repeat(batch_size, 1, 1, 1))
            pred_logit = model(photo, sketch)
            pred_logits.append(pred_logit)

        pred_logits = torch.cat(pred_logits, dim=1)
        import pdb;pdb.set_trace()
        pred = F.softmax(pred_logits, dim=1)
        pred = torch.max(pred_logits, dim=1)[1]

        accuracy = torch.sum(pred == label.long()) / batch_size
        acc_meter.update(accuracy.data[0])

        pbar.update()
    pbar.close()

    print('Accuracy: %f' % acc_meter.avg)
