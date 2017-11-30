from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cPickle
import numpy as np

import torch
from torch.autograd import Variable

from datasets import ContextFreePreloadedGenerator as Generator
from datasets import INSTANCE_NAME2IX_DICT, CATEGORY_NAME2IX_DICT
from datasets import INSTANCE_IX2NAME_DICT, CATEGORY_IX2NAME_DICT
from datasets import CATEGORY_TO_INSTANCE_DICT
from model import load_checkpoint


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to where model is stored')
    parser.add_argument('--layer', type=str, help='conv_4_2|fc7', default='conv_4_2')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.layer in ['conv_4_2', 'fc7']

    data_dir = '/data/jefan/sketchpad_basic_fixedpose96_conv_4_2'
    with open(os.path.join(data_dir, 'preloaded_context_all.pkl'), 'r') as fp:
        data = cPickle.load(fp)
        cat2target = data['cat2target']
        target2sketch = data['target2sketch']
        target2condition = data['target2condition']

    instance2closesketch = {}
    instance2farsketch = {}
    instance2firsttarget = {}

    sketch_paths = []
    for target, sketch in target2sketch.iteritems():
        instance = os.path.splitext(target)[0].split('_')[-1]
        if target2condition[target] == 'closer':
            instance2closesketch[instance] = sketch
        elif target2condition[target] == 'further':
            instance2farsketch[instance] = sketch
        sketch_paths.append(sketch)
        instance2firsttarget[instance] = target

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()

    render_features = np.zeros((32, 1000))
    sketch_close_features = np.zeros((32, 1000))
    sketch_far_features = np.zeros((32, 1000))

    # loop through each of our targets first
    for i in xrange(4):
        instances = CATEGORY_TO_INSTANCE_DICT[CATEGORY_IX2NAME_DICT[i]]
        for j in xrange(8):
            target_path = os.path.join(data_dir, 'target', instance2firsttarget[instances[j]])
            target = torch.from_numpy(np.load(target_path))
            if args.cuda:
                target = target.cuda()
            target = Variable(target, requires_grad=False)
            target_embedding = model.photo_adaptor(target.unsqueeze(0))
            target_embedding = target_embedding.squeeze(0).cpu().data.numpy()
            render_features[i * 8 + j] = target_embedding

    # loop through each of our close sketches for each target and we will need
    # to average them over.
    for i in xrange(4):
        instances = CATEGORY_TO_INSTANCE_DICT[CATEGORY_IX2NAME_DICT[i]]
        for j in xrange(8):
            close_sketches = instance2closesketch[instances[j]]
            n_close_sketches = len(close_sketches)
            close_batch = np.zeros(n_close_sketches, 512, 28, 28)

            for k in xrange(n_close_sketches):
                sketch_path = os.path.join(data_dir, 'sketch', close_sketches[k])
                close_batch[k] = np.load(sketch_path)

            close_batch = torch.from_numpy(close_batch)
            if args.cuda:
                close_batch = close_batch.cuda()
            close_batch = Variable(close_batch, requires_grad=False)
            close_batch_embedding = model.sketch_adaptor(close_batch)
            close_batch_embedding = torch.mean(close_batch_embedding, dim=0)

            close_batch_embedding = close_batch_embedding.cpu().data.numpy()
            sketch_close_features[i * 8 + j] = close_batch_embedding

    # loop through each of our far sketches for each target and we will need 
    # to average them over.
    for i in xrange(4):
        instances = CATEGORY_TO_INSTANCE_DICT[CATEGORY_IX2NAME_DICT[i]]
        for j in xrange(8):
            far_sketches = instance2farsketch[instances[j]]
            n_far_sketches = len(far_sketches)
            far_batch = np.zeros(n_far_sketches, 512, 28, 28)

            for k in xrange(n_far_sketches):
                sketch_path = os.path.join(data_dir, 'sketch', far_sketches[k])
                far_batch[k] = np.load(sketch_path)

            far_batch = torch.from_numpy(far_batch)
            if args.cuda:
                far_batch = far_batch.cuda()
            far_batch = Variable(far_batch, requires_grad=False)
            far_batch_embedding = model.sketch_adaptor(far_batch)
            far_batch_embedding = torch.mean(far_batch_embedding, dim=0)

            far_batch_embedding = far_batch_embedding.cpu().data.numpy()
            sketch_far_features[i * 8 + j] = far_batch_embedding

    features = np.concatenate((render_features, 
                               sketch_close_features, 
                               sketch_far_features), axis=0)
    rdm = np.corrcoef(features)

    plt.figure()
    plt.tight_layout()
    
    

