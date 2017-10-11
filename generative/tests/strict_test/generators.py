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

SAME_PHOTO_EX = 0
SAME_CLASS_EX = 1


class EasyGenerator(object):
    """This will generate data according to the following pairs:
        
        (photo1, sketch1), 
        (photo2, sketch2), 
        (photo1, sketch2), 
        (photo2, sketch1)

    where photo2 is a photo of the same class.
    Each batch will be composed of a number of these 4-pair structs.

    Notably, across different epochs, repeated calls to this 
    generator will return different photo1, photo2 which should help
    with generalization.

    The train-test split here is made up of intra-category splits
    i.e. 80% of dog photos are used in training, and 20% of dog
    photos are used in testing.

    :param photo_emb_dir: pass to photo embedding directories.
    :param sketch_emb_dir: pass to sketch embedding directories.
    :param batch_size: number of 4-pair structs to return at a time.
    :param use_cuda: if True, make CUDA compatible objects.
    :param train: decides which split of data to sample from.
    """
    def __init__(self, photo_emb_dir, sketch_emb_dir, train=True,
                 batch_size=10, use_cuda=False):
        self.photo_emb_dir = photo_emb_dir
        self.sketch_emb_dir = sketch_emb_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.train = train
        self.size = self.get_size()

    def train_test_split(self):
        """Returns the photos to be used in training and the photos to 
        be used in testing. The photos will be randomly sorted.
        """
        categories = glob(os.path.join(self.photo_emb_dir, '*'))
        train_photos, test_photos = [], []
        for cat in categories:
            paths = glob(os.path.join(cat, '*'))
            split = int(0.8 * len(paths))
            train_photos += paths[:split]
            test_photos += paths[split:]

        random.shuffle(train_photos)
        random.shuffle(test_photos)

        return train_photos, test_photos

    def get_size(self):
        train, test = self.train_test_split()
        return len(train) if self.train else len(test)

    def make_generator(self):
        dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        
        # get a list of photo paths that we will loop through
        train_photo_paths, test_photo_paths = self.train_test_split()
        photo_paths = train_photo_paths if self.train else test_photo_paths

        # keep track of when to start new batch
        batch_idx = 0

        for i in range(self.size):
            photo1_path = photo_paths[i]
            sketch1_path = sample_sketch_from_photo_path(photo1_path, self.sketch_emb_dir)
            photo2_path = sample_photo_from_photo_path(photo1_path, self.photo_emb_dir, 
                                                       same_class=True)
            sketch2_path = sample_sketch_from_photo_path(photo2_path, self.sketch_emb_dir)

            # load all of these into numpy
            photo1 = np.load(photo1_path)
            sketch1 = np.load(sketch1_path)
            photo2 = np.load(photo2_path)
            sketch2 = np.load(sketch2_path)

            photo_group = np.vstack((photo1, photo2, photo1, photo2))
            sketch_group = np.vstack((sketch1, sketch2, sketch1, sketch2))
            label_group = np.array((1, 1, 0, 0))

            if batch_idx == 0:
                photo_batch = photo_group
                sketch_batch = sketch_group
                label_batch = label_group
            else:
                photo_batch = np.vstack((photo_batch, photo_group))
                sketch_batch = np.vstack((sketch_batch, sketch_group))
                label_batch = np.concatenate((label_batch, label_group))

            batch_idx += 1

            if batch_idx == self.batch_size:
                photo_batch = Variable(torch.from_numpy(photo_batch)).type(dtype)
                sketch_batch = Variable(torch.from_numpy(sketch_batch)).type(dtype)
                label_batch = Variable(torch.from_numpy(label_batch), 
                                       requires_grad=False).type(dtype)

                yield (photo_batch, sketch_batch, label_batch)
                batch_idx = 0

        if batch_idx > 0:
            photo_batch = Variable(torch.from_numpy(photo_batch)).type(dtype)
            sketch_batch = Variable(torch.from_numpy(sketch_batch)).type(dtype)
            label_batch = Variable(torch.from_numpy(label_batch), 
                                   requires_grad=False).type(dtype)

            yield (photo_batch, sketch_batch, label_batch)


class HardGenerator(EasyGenerator):
    """See EasyGenerator for a detailed explanation.

    The distinction here is that the train-test split will be made
    up of inter-category splits i.e 80% of categories are used in
    training, and 20% of categories are used in testing.
    """
    def train_test_split(self):
        """Returns the photos to be used in training and the photos to 
        be used in testing. The photos will be randomly sorted.
        """
        categories = glob(os.path.join(self.photo_emb_dir, '*'))
        split = int(0.8 * len(categories))

        train_photos, test_photos = [], []
        for i, cat in enumerate(categories):
            paths = glob(os.path.join(cat, '*'))
            if i < split:
                train_photos += paths
            else:
                test_photos += paths

        random.shuffle(train_photos)
        random.shuffle(test_photos)

        return train_photos, test_photos


def sample_sketch_from_photo_path(photo_path, sketch_emb_dir):
    photo_name, photo_ext = os.splitext(os.path.basename(photo_path))
    photo_folder = os.path.dirname(photo_path).split('/')[-1]

    sketch_paths = glob(os.path.join(sketch_emb_dir, photo_folder, 
                        '{name}-*{ext}'.format(name=photo_name, ext=photo_ext)))
    sketch_path = np.random.choice(sketch_paths)
    return sketch_path


def sample_photo_from_photo_path(photo_path, photo_emb_dir, same_class=False):
    photo_name, photo_ext = os.splitext(os.path.basename(photo_path))
    photo_folder = os.path.dirname(photo_path).split('/')[-1]

    if same_class:
        photo_paths = glob(os.path.join(photo_emb_dir, photo_folder, '*'))
        photo_paths.remove(photo_path)
    else:
        categories = glob(os.path.join(self.photo_emb_dir, '*'))
        categories.remove(photo_folder)
        category = np.random.choice(categories)
        photo_paths = glob(os.path.join(photo_emb_dir, category, '*'))

    return np.random.choice(photo_paths)
