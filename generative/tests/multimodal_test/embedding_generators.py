"""Like distribution_test/generators.py (which operates on raw images), but 
this operates on pre-computed embeddings.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from glob import glob
import numpy as np

import torch
from torch.autograd import Variable


def generator(data_type, batch_size=32, use_cuda=False):
    """Returns (image, sketch) pairs where the sketch is of the image..

    :param data_type: data|noisy|swapped|neighbor
    """
    assert data_type in ['data', 'noisy', 'swapped', 'neighbor']

    photo_emb_dir = '/home/wumike/full_sketchy_embeddings/photos'
    if data_type == 'noisy':
        sketch_emb_dir = '/home/wumike/full_sketchy_embeddings/noise'
    else:
        sketch_emb_dir = '/home/wumike/full_sketchy_embeddings/sketches'

    categories = os.listdir(sketch_emb_dir)
    photo_paths = [path for path in list_files(photo_emb_dir, ext='npy') 
                   if os.path.dirname(path).split('/')[-1] in categories]
    sketch_paths = [path for path in list_files(sketch_emb_dir, ext='npy') 
                   if os.path.dirname(path).split('/')[-1] in categories]
    n_paths = len(sketch_paths)
    photo_batch = None
    sketch_batch = None

    for i in range(n_paths):
        # depending on the type of generator we want, we will load 
        # sketches and matching photos differently.
        if data_type in ['data', 'noisy']:
            sketch_path = sketch_paths[i]
            sketch_filename = os.path.basename(sketch_path)
            sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
            photo_filename = sketch_filename.split('-')[0] + '.npy'
            photo_path = os.path.join(photo_emb_dir, sketch_folder, photo_filename)
        elif data_type == 'swapped':
            sketch_path = sketch_paths[i]
            sketch_filename = os.path.basename(sketch_path)
            sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
            while True:
                photo_path = np.random.choice(photo_paths)
                photo_folder = os.path.dirname(photo_path).split('/')[-1]
                if photo_folder != sketch_folder:
                    break
        elif data_type == 'neighbor':
            sketch_path = sketch_paths[i]
            sketch_filename = os.path.basename(sketch_path)
            sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
            matching_photo_filename = sketch_filename.split('-')[0] + '.npy'
            matching_photo_path = os.path.join(photo_emb_dir, sketch_folder, 
                                               matching_photo_filename)             
            matching_photo_folder = os.path.dirname(matching_photo_path)

            while True:                        
                photo_filename = np.random.choice(os.listdir(matching_photo_folder))
                photo_path = os.path.join(matching_photo_folder, photo_filename)
                if photo_filename != matching_photo_filename:
                    break

        # load the embeddings and return them as batched PyTorch objects
        photo = np.load(photo_path)
        sketch = np.load(sketch_path)

        if photo_batch is None and sketch_batch is None:
            photo_batch = photo
            sketch_batch = sketch
        else:
            photo_batch = np.vstack((photo_batch, photo))
            sketch_batch = np.vstack((sketch_batch, sketch))

        if photo_batch.shape[0] == batch_size:
            photo_batch = torch.from_numpy(photo_batch)
            sketch_batch = torch.from_numpy(sketch_batch)

            photo_batch = Variable(photo_batch, volatile=True)
            sketch_batch = Variable(sketch_batch, volatile=True)

            if use_cuda:
                photo_batch = photo_batch.cuda()
                sketch_batch = sketch_batch.cuda()

            yield (photo_batch, sketch_batch)
            
            photo_batch = None
            sketch_batch = None

    # return any remaining data
    if photo_batch is not None and sketch_batch is not None:
        photo_batch = torch.from_numpy(photo_batch)
        sketch_batch = torch.from_numpy(sketch_batch)

        photo_batch = Variable(photo_batch, volatile=True)
        sketch_batch = Variable(sketch_batch, volatile=True)

        if use_cuda:
            photo_batch = photo_batch.cuda()
            sketch_batch = sketch_batch.cuda()

        yield (photo_batch, sketch_batch)


def list_files(path, ext='jpg'):
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result
