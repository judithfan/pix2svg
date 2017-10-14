"""Given a trained multimodal model, plot the distributions for
each of our (photo, sketch) pairs. We can do this for the 
training and testing sets. We will also be evaluating against 
(photo, noise) pairs.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np

import torch
from torch.autograd import Variable

# load internal functions
from convmodel import cosine_similarity
from convmodel import load_checkpoint
from generators import EasyApplyGenerator
from generators import HardApplyGenerator


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_dir', type=str)
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('noise_emb_dir', type=str)
    parser.add_argument('distance_path', type=str, help='where to save distances.')
    parser.add_argument('model_path', type=str, help='where to find trained model.')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--hard', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    checkpoint = torch.load(args.model_path)

    model.eval()
    if args.cuda:
        model.cuda()

    ApplyGenerator = HardApplyGenerator if args.hard else EasyApplyGenerator
    generator = ApplyGenerator(args.photo_emb_dir, args.sketch_emb_dir, 
                               args.noise_emb_dir, train=args.train, use_cuda=args.cuda)
    examples = generator.make_generator()
    count = 0  # track number of examples seen
    distances = np.zeros((generator.size, 2))  # store distances between test examples here
    while True:
        try:
            photo, sketch, _, _, pairtype = examples.next()
        except StopIteration:
            break
       
        photo = model.photo_adaptor(photo)
        sketch = model.sketch_adaptor(sketch)

        # compute pearson correlation
        photo = photo - torch.mean(photo, dim=1, keepdim=True)
        sketch = sketch - torch.mean(sketch, dim=1, keepdim=True)
        dist = cosine_similarity(photo, sketch, dim=1)
        dist_np = float(dist.cpu().data.numpy()[0])

        distances[count, 0] = dist_np
        distances[count, 1] = pairtype

        count += 1
        print('Compute Distance [{}/{}].'.format(count, generator.size))

    np.save(args.distance_path, distances)
    print('Distances saved to %s.' % args.distance_path)
