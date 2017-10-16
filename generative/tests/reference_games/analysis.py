from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import json
import torch

from generators import ReferenceGameEmbeddingGenerator

sys.path.append('..')
from strict_test.convmodel import load_checkpoint
from strict_test.convmodel import cosine_similarity


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sketch_emb_dir', type=str, help='path to sketches')
    parser.add_argument('render_emb_dir', type=str, help='path to renderings')
    parser.add_argument('json_path', type=str, help='path to where to dump json')
    parser.add_argument('model_dir', type=str, help='path to trained model')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # define data generator
    generator = ReferenceGameEmbeddingGenerator(args.sketch_emb_dir, args.render_emb_dir, 
                                                use_cuda=args.cuda)
    examples = generator.make_generator() 
    print('Built generator.')

    # define strict multimodal model
    model = load_checkpoint(args.model_dir, use_cuda=args.cuda)
    model.eval()
    if model.cuda:
        model.cuda()
    print('Loaded model.')

    count = 0  # track number of examples seen
    results = []
   
    while True:
        try:  # exhaust the generator to return all pairs. this loads 1 tuple at a time.
            sketch_path, sketch, render_path, render = examples.next()
        except StopIteration:
            break
       
        pred_proba = model(render, sketch)
        pred_proba = float(pred_proba.cpu().data.numpy()[0])
        label = pairtype == 0  # only (photo, matching sketch) is a positive example

        r = {'render': render_path,
             'sketch': sketch_path,
             'proba': pred_proba,
             'label': label}
        results.append(r)

        count += 1
        print('Compute prediction [{}/{}].'.format(count, generator.size))

    with open(args.json_path, 'w') as fp:
        json.dump(results, fp)
