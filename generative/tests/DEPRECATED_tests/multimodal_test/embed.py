"""Given a trained multimodal model, generate embeddings."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
from glob import glob

import torch
from torch.autograd import Variable
from model import load_checkpoint


def mimic_dir(new_dir, old_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    # check and create all sub-directories
    all_folders = [x[0] for x in os.walk(old_dir)]
    for folder in all_folders:
        new_folder = folder.replace(old_dir, new_dir)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_dir', type=str)
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('photo_mmemb_dir', type=str)
    parser.add_argument('sketch_mmemb_dir', type=str)
    parser.add_argument('model_path', type=str, help='where to find trained model.')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    # load trained multimodal model
    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()

    # create new directories if not existing.
    mimic_dir(args.photo_mmemb_dir, args.photo_emb_dir)
    mimic_dir(args.sketch_mmemb_dir, args.sketch_emb_dir)

    photo_paths = glob(os.path.join(args.photo_emb_dir, '*/*.npy'))
    sketch_paths = glob(os.path.join(args.sketch_emb_dir, '*/*.npy'))
    
    n_photos = len(photo_paths)
    n_sketches = len(sketch_paths)

    for i in xrange(n_photos):
        photo = np.load(photo_paths[i])
        photo = Variable(torch.from_numpy(photo).type(dtype), volatile=True)
        photo = model.photo_adaptor(photo.unsqueeze(0))
        photo = photo.cpu().data.numpy()[0]
        np.save(photo_paths[i].replace(args.photo_emb_dir, args.photo_mmemb_dir), photo)
        print('Compute Photo Embedding [{}/{}].'.format(i + 1, n_photos))


    for i in xrange(n_sketches):
        sketch = np.load(sketch_paths[i])
        sketch = Variable(torch.from_numpy(sketch).type(dtype), volatile=True)
        sketch = model.sketch_adaptor(sketch.unsqueeze(0))
        sketch = sketch.cpu().data.numpy()[0]
        np.save(sketch_paths[i].replace(args.sketch_emb_dir, args.sketch_mmemb_dir), sketch)
        print('Compute Sketch Embedding [{}/{}].'.format(i + 1, n_sketches))

