"""multimodaltest.py trains a "Multi-modal" model where natural photo 
and sketch are 2 modalities that get projected into a shared space.
We can then use this shared space to compare if the different kinds of
distributions are dragged apart.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
sys.path.append('../distribution_test')

import copy
from glob import glob
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms
from transformtest import load_checkpoint

from distribtest import gen_distance
from embedding_generator import generator


class MultiModalLayerLossTest(BaseLossTest):
    """Because we pre-computed embeddings for our entire dataset of 
    natural photos and sketches, this LossTest is a little different 
    than others: we don't need to load VGG and can directly operate
    on the saved embeddings.
    """

    def __init__(self, model_path, metric='euclidean', use_cuda=False):
        super(MultiModalLayerLossTest, self).__init__()
        model = load_checkpoint(model_path, use_cuda=use_cuda)
        mode.eval()

        if use_cuda:
            model.cuda()
        
        self.model = model
        self.metric = metric
        self.use_cuda = use_cuda

    def loss(self, image_embs, sketch_embs):
        image_embs = self.model.photo_adaptor(image_embs)
        sketch_embs = self.model.sketch_adaptor(sketch_embs)
        return gen_distance(images_emb, sketches_emb, metric=self.metric)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('out_folder', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_type', type=str, default='data')
    args = parser.parse_args()

    assert args.data_type in ['data', 'noisy', 'swapped', 'neighbor']

    use_cuda = torch.cuda.is_available()
    generator = generator(args.data_type, batch_size=args.batch_size, use_cuda=use_cuda)
    layer_test = MultiModalLayerLossTest(args.model_path, use_cuda=use_cuda)

    batch_idx = 0
    example_cnt = 0

    while True:
        try:
            photo_batch, sketch_batch = generator.next()
            batch_idx += 1
            example_cnt += photo_batch.size()[0]
        except StopIteration:
            break

        losses = layer_test.loss(photo_batch, sketch_batch)
        losses = losses.cpu().data.numpy().flatten()
        loss_list += losses.tolist()

        print('Batch {} | Examples {}'.format(batch_index, example_cnt))

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    loss_list = np.array(loss_list)
    filename = 'loss_multimodal_{datatype}.npy'.format(datatype=args.data_type)
    np.save(os.path.join(args.out_folder, filename), loss_list)
