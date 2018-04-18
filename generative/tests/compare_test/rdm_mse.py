from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.autograd import Variable
from dataset import ExhaustiveDataset
from train_mse import load_checkpoint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if model.cuda:
        model.cuda()

    dataset = ExhaustiveDataset(layer='conv42')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    object_order = dataset.object_order

    rdm_further_sums = np.zeros((32, 32))
    rdm_closer_sums  = np.zeros((32, 32))
    rdm_further_cnts = np.zeros((32, 32))
    rdm_closer_cnts = np.zeros((32, 32))

    pbar = tqdm(total=len(loader))
    for batch_idx, (sketch, sketch_object, sketch_context, sketch_path) in enumerate(loader):
        sketch = Variable(sketch, volatile=True)
        sketch_object_ix = object_order.index(sketch_object[0])
        if args.cuda:
            sketch = sketch.cuda()
        photo_generator = dataset.gen_photos()
        for photo, photo_object, photo_path in photo_generator():
            photo = Variable(photo, volatile=True)
            photo_object_ix = object_order.index(photo_object)
            batch_size = len(sketch)
            if args.cuda:
                photo = photo.cuda()
            pred, _, _ = model(photo, sketch)
            pred = pred.squeeze(1).cpu().data[0]
            if sketch_context[0] == 'closer':
                rdm_closer_sums[photo_object_ix, sketch_object_ix] += pred
                rdm_closer_cnts[photo_object_ix, sketch_object_ix] += 1
            elif sketch_context[0] == 'further':
                rdm_further_sums[photo_object_ix, sketch_object_ix] += pred
                rdm_further_cnts[photo_object_ix, sketch_object_ix] += 1
            else:
                raise Exception('Unrecognized context: %s.' % sketch_context[0])
        pbar.update()
    pbar.close()

    rdm_further = rdm_further_sums / rdm_further_cnts
    rdm_closer = rdm_closer_sums / rdm_closer_cnts

    import seaborn as sns; sns.set()
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    plt.figure()
    ax = sns.heatmap(rdm_further, linewidths=.5)
    fig = ax.get_figure()
    fig.savefig('./rdm-further.pdf')

    plt.figure()
    ax = sns.heatmap(rdm_closer, linewidths=.5)
    fig = ax.get_figure()
    fig.savefig('./rdm-closer.pdf')

    plt.figure()
    ax = sns.heatmap(rdm_closer - rdm_further, linewidths=.5)
    fig = ax.get_figure()
    fig.savefig('./rdm-diff.pdf')
