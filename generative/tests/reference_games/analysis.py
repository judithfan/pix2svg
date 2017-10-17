from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import csv
import json
import torch

from generators import ReferenceGameEmbeddingGenerator

sys.path.append('..')
from strict_test.convmodel import load_checkpoint
from strict_test.convmodel import cosine_similarity


def sketch_to_render_dict():
    data = []
    with open('./data/sketchpad_basic_merged_group_data.csv') as fp:
        reader = csv.reader(fp)
        for row in reader:
            data.append(row)

    header = data[0]
    data = data[1:]

    gameID_ix = header.index('gameID')
    trialNum_ix = header.index('trialNum')
    target_ix = header.index('target')
    Distractor1_ix = header.index('Distractor1')
    Distractor2_ix = header.index('Distractor2')
    Distractor3_ix = header.index('Distractor3')
    pose_ix = header.index('pose')

    lookup = {}

    for row in data:
        sketch_name = 'gameID_{id}_trial_{trial}.npy'.format(
            id=row[gameID_ix], trial=row[trialNum_ix])
        if row[pose_ix] != 'NA':
            target_name = '{cat}_{id:04d}.npy'.format(
                cat=row[target_ix], id=int(row[pose_ix]))
            distractor_names = [
                '{cat}_{id:04d}.npy'.format(cat=row[Distractor1_ix], id=int(row[pose_ix])),
                '{cat}_{id:04d}.npy'.format(cat=row[Distractor2_ix], id=int(row[pose_ix])),
                '{cat}_{id:04d}.npy'.format(cat=row[Distractor3_ix], id=int(row[pose_ix])),
            ]
            lookup[sketch_name] = (target_name, distractor_names)
        else:
            lookup[sketch_name] = None

    return lookup


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

    # define lookup
    match_lookup = sketch_to_render_dict()

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

        sketch_basename = os.path.basename(sketch_path)
        render_basename = os.path.basename(render_path)
      
        if sketch_basename not in match_lookup:
            print('Bad sketch name found: %s.' % sketch_basename)
            continue

        lookup_row = match_lookup[sketch_basename]
        if lookup_row is None:
            print('Bad sketch name found: %s.' % sketch_basename)
            continue
        
        if '_'.join(render_basename.split('_')[2:]) == lookup_row[0]:
            label = 1
        # elif '_'.join(render_basename.split('_')[2:]) in lookup_row[1]:
        #     label = 0
        else:
            label = 0
            # raise Exception('what... lookup failed.')

        pred_proba = model(render, sketch)
        pred_proba = float(pred_proba.cpu().data.numpy()[0])

        r = {'render': render_path,
             'sketch': sketch_path,
             'proba': pred_proba,
             'label': label}
        results.append(r)

        count += 1
        print('Compute prediction [{}/{}].'.format(count, generator.size))

    with open(args.json_path, 'w') as fp:
        json.dump(results, fp)
