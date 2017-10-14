"""What are some examples of (photo, sketch) pairs that are 
scoring extremely low or extremely high.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import json
import numpy as np

import torch
from torch.autograd import Variable

# load internal functions
from model import cosine_similarity
from convmodel import load_checkpoint
from generators import EasyApplyGenerator
from generators import HardApplyGenerator


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_dir', type=str)
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('noise_emb_dir', type=str)
    parser.add_argument('json_path', type=str, help='where to save probabilities.')
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
    
    results = []
   
    while True:
        try:
            photo, sketch, photo_path, sketch_path, pairtype = examples.next()
        except StopIteration:
            break
       
        pred_proba = model(photo, sketch)
        pred_proba = float(pred_proba.cpu().data.numpy()[0])
        label = pairtype == 0  # only (photo, matching sketch) is a positive example

        r = {'photo': photo_path,
             'sketch': sketch_path,
             'proba': pred_proba,
             'label': label}
        results.append(r)

        count += 1
        print('Compute prediction [{}/{}].'.format(count, generator.size))

    with open(args.json_path, 'w') as fp:
        json.dump(results, fp)
