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
from model import cosine_similarity
from model import load_checkpoint
from generators import ApplyGenerator


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_dir', type=str)
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('noise_emb_dir', type=str)
    parser.add_argument('distance_path', type=str, help='where to save distances.')
    parser.add_argument('model_path', type=str, help='where to find trained model.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    checkpoint = torch.load(args.model_path)

    model.eval()
    if args.cuda:
        model.cuda()

    generator = ApplyGenerator(args.photo_emb_dir, args.sketch_emb_dir, 
                               args.noise_emb_dir, batch_size=args.batch_size,  
                               train=args.train, use_cuda=args.cuda)
    examples = generator.make_generator()
    count = 0  # track number of examples seen
    distances = np.zeros((generator.size, 2))  # store distances between test examples here
    while True:
        try:
            photos, sketches = examples.next()
            examples_size = len(photos)
        except StopIteration:
            break
       
        photos = model.photo_adaptor(photos)
        sketches = model.sketch_adaptor(sketches)

        # compute pearson correlation
        photos = photos - torch.mean(photos, dim=1, keepdim=True)
        sketches = sketches - torch.mean(sketches, dim=1, keepdim=True)
        dists = cosine_similarity(photos, sketches, dim=1)
        
        dists_np = float(dists.cpu().data.numpy()[0])

        distances[count:count + examples_size, 0] = dists_np
        distances[count:count + examples_size, 1] = dists_type

        count += examples_size
        print('Compute Distance [{}/{}].'.format(count, generator.size))

    np.save(args.distance_path, distances)
    print('Distances saved to %s.' % args.distance_path)
