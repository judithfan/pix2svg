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
from referenceutils import ThreeClassGenerator, FourClassGenerator, PoseGenerator


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('generator', type=str, help='cross|intra|pose')
    parser.add_argument('json_path', type=str, help='where to save probabilities.')
    parser.add_argument('model_path', type=str, help='where to find trained model.')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    assert args.generator in ['cross', 'intra', 'pose']
    if args.generator == 'cross':
        ReferenceGenerator = ThreeClassGenerator
    elif args.generator == 'intra':
        ReferenceGenerator = FourClassGenerator
    elif args.generator == 'pose':
        ReferenceGenerator = PoseGenerator
    else:
        raise Exception('How did you get here?')

    render_emb_dir = '/data/jefan/sketchpad_basic_extract/subordinate_allrotations_6_minified_conv_4_2'
    sketch_emb_dir = '/data/jefan/sketchpad_basic_extract/sketch_conv_4_2/'

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()

    generator = ReferenceGenerator(render_emb_dir, sketch_emb_dir, batch_size=1, 
                                   train=args.train, use_cuda=args.cuda, return_paths=True)
    examples = generator.make_generator()
    count = 0  # track number of examples seen
    
    results = []
    while True:
        try:
            render, render_path, sketch, sketch_path, label = examples.next()
        except StopIteration:
            break
       
        pred_proba = model(render, sketch).cpu().data.squeeze(1)
        label = label.cpu().data

        for i in range(4):
            r = {'render': render_path[i],
                 'sketch': sketch_path[i],
                 'proba': pred_proba[i],
                 'label': label[i]}
            results.append(r)

        count += 1
        print('Compute prediction [{}/{}].'.format(count, generator.size))

    with open(args.json_path, 'w') as fp:
        json.dump(results, fp)
