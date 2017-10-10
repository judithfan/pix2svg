from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import random
from glob import glob

import numpy as np
import torch
from torch.autograd import Variable


class L2TrainGenerator(object):
    def __init__(self, photo_emb_dir, sketch_emb_dir, train=True,
                 batch_size=32, use_cuda=False):
        self.photo_emb_dir = photo_emb_dir
        self.sketch_emb_dir = sketch_emb_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.train = train
        self.size = self.get_size()

    def get_size(self):
        # this function is used to get the number of images that will be returned
        # via this generator.
        categories = os.listdir(self.sketch_emb_dir)
        n_categories = len(categories)
        if self.train:
            categories = categories[:int(n_categories * 0.8)]
        else:
            categories = categories[int(n_categories * 0.8):]
        sketch_paths = [path for path in list_files(self.sketch_emb_dir, ext='npy') 
                        if os.path.dirname(path).split('/')[-1] in categories]
        return len(sketch_paths)

    def make_generator(self):
        dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # note that everytime this is called, it will return slightly different
        # pairs since not all pairs are used in 1 epoch. But it will never
        # intersect with the test directory.
        categories = os.listdir(self.sketch_emb_dir)
        n_categories = len(categories)

        # test is composed of sketches from different categories. This 
        # clearly distinguishes which sketches and photos are being used
        # to train and which are for testing.
        if self.train:
            categories = categories[:int(n_categories * 0.8)]
        else:
            categories = categories[int(n_categories * 0.8):]
        photo_paths = [path for path in list_files(self.photo_emb_dir, ext='npy') 
                       if os.path.dirname(path).split('/')[-1] in categories]
        sketch_paths = [path for path in list_files(self.sketch_emb_dir, ext='npy') 
                       if os.path.dirname(path).split('/')[-1] in categories]

        # we don't want class imbalance: in one epoch, we will show each
        # sketch once. This sketch could be used in a same_photo pair, 
        # same_class pair or diff_class pair. Again, over different epochs, 
        # randomness will lead to a sketch / photo being used in different
        # pairs which should be a form of regularization itself.
        random.shuffle(sketch_paths)
        n_paths = len(sketch_paths)

        # None is how we keep track of batches.
        photo_batch = None 
        sketch_batch = None 

        for i in range(n_paths):
            # for this generator, we actually only need positive examples 
            # of (photo, sketch).
            sketch_path = sketch_paths[i]
            photo_path = get_photo_from_sketch_path(sketch_path, self.photo_emb_dir)
    
            photo = np.load(photo_path)
            sketch = np.load(sketch_path)

            if photo_batch is None:
                photo_batch = photo[np.newaxis, ...]
                sketch_batch = sketch[np.newaxis, ...]
            else:
                photo_batch = np.vstack((photo_batch, photo))
                sketch_batch = np.vstack((sketch_batch, sketch))

            if photo_batch.shape[0] == self.batch_size:
                # this is for training, so volatile is False
                photo_batch = Variable(torch.from_numpy(photo_batch).type(dtype),
                                       volatile=not self.train)
                sketch_batch = Variable(torch.from_numpy(sketch_batch).type(dtype),
                                        volatile=not self.train)

                # yield data so we can continue to query the same object
                yield (photo_batch, sketch_batch)
                photo_batch = None
                sketch_batch = None 

        # return any remaining data
        # since n_data may not be divisible by batch_size
        if photo_batch is not None:
            photo_batch = Variable(torch.from_numpy(photo_batch).type(dtype),
                                   volatile=not self.train)
            sketch_batch = Variable(torch.from_numpy(sketch_batch).type(dtype),
                                    volatile=not self.train)

            yield (photo_batch, sketch_batch)
            photo_batch = None
            sketch_batch = None 
