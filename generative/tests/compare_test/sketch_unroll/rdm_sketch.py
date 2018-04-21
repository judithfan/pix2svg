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
import torch.nn.functional as F
from torch.autograd import Variable

from dataset_sketch import ExhaustiveSketchDataset
from train_sketch import load_checkpoint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model')
    parser.add_argument('--batch-size', type=int, default=64, help='size of minibatch [default: 64]')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if model.cuda:
        model.cuda()

    dataset = ExhaustiveSketchDataset(layer='conv42', split='full')
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    object_order = dataset.object_order

    rdm_further_sums = np.zeros((32, 32))
    rdm_closer_sums  = np.zeros((32, 32))
    rdm_further_cnts = np.zeros(32)
    rdm_closer_cnts = np.zeros(32)

    pbar = tqdm(total=len(loader))
    for batch_idx, (sketch, sketch_object, sketch_context, sketch_path) in enumerate(loader):
        batch_size = len(sketch)
        sketch = Variable(sketch, volatile=True)
        sketch_object_ix = [object_order.index(sketch_object[i]) for i in xrange(batch_size)]
        if args.cuda:
            sketch = sketch.cuda()
        
        pred_logits = model(sketch)
        pred = F.softmax(pred_logits, dim=1)
        pred = pred.cpu().data.numpy()

	for t in xrange(batch_size):
            if sketch_context[0] == 'closer':
                rdm_closer_sums[:, sketch_object_ix[t]] += pred[t]
                rdm_closer_cnts[sketch_object_ix[t]] += 1
            elif sketch_context[0] == 'further':
                rdm_further_sums[:, sketch_object_ix[t]] += pred[t]
                rdm_further_cnts[sketch_object_ix[t]] += 1
            else:
                raise Exception('Unrecognized context: %s.' % sketch_context[0])
        pbar.update()
    pbar.close()

    for i in xrange(32):
        rdm_further_sums[:, i] /= rdm_further_cnts[i]
        rdm_closer_sums[:, i] /= rdm_closer_cnts[i]

    import seaborn as sns; sns.set()
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    plt.figure()
    ax = sns.heatmap(rdm_further_sums, linewidths=.5)
    fig = ax.get_figure()
    fig.savefig('./rdm-further.pdf')

    plt.figure()
    ax = sns.heatmap(rdm_closer_sums, linewidths=.5)
    fig = ax.get_figure()
    fig.savefig('./rdm-closer.pdf')

    plt.figure()
    ax = sns.heatmap(rdm_closer_sums - rdm_further_sums, linewidths=.5)
    fig = ax.get_figure()
    fig.savefig('./rdm-diff.pdf')
