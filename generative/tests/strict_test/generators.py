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
DIFF_CLASS_EX = 2
NOISE_EX = 3


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
        blacklist_paths = test_photo_paths if self.train else train_photo_paths

        # keep track of when to start new batch
        batch_idx = 0

        for i in range(self.size):
            photo1_path = photo_paths[i]
            sketch1_path = sample_sketch_from_photo_path(photo1_path, self.sketch_emb_dir,
                                                         deterministic=False)
            photo2_path = sample_photo_from_photo_path(photo1_path, self.photo_emb_dir, 
                                                       blacklist=blacklist_paths, same_class=True)
            sketch2_path = sample_sketch_from_photo_path(photo2_path, self.sketch_emb_dir,
                                                         deterministic=False)

            # load all of these into numpy
            photo1 = np.load(photo1_path)[np.newaxis, ...]
            sketch1 = np.load(sketch1_path)[np.newaxis, ...]
            photo2 = np.load(photo2_path)[np.newaxis, ...]
            sketch2 = np.load(sketch2_path)[np.newaxis, ...]

            photo_group = np.vstack((photo1, photo2, photo1, photo2))
            sketch_group = np.vstack((sketch1, sketch2, sketch2, sketch1))
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


class ApplyGenerator(object):
    """This data generator returns (photo, sketch) and is meant to be used
    only for inference. This is distinct than passing `train=False` as a 
    parameter because it returns noise and other types of (photo, sketch) pairs.

    Critically, this is distinct from MultiModalApplyGenerator because here we 
    do train-test split within category, not across.

    :param photo_emb_dir: pass to photo embedding directories
    :param sketch_emb_dir: pass to sketch embedding directories
    :param noise_emb_dir: mix in noise samples in negatives.
    :param use_cuda: if True, make CUDA compatible object.
    :param train: This is only used for inference but if train=True, then we can 
                  do inference on the training set.
    """
    def __init__(self, photo_emb_dir, sketch_emb_dir, noise_emb_dir,
                 train=False, use_cuda=False):
        self.photo_emb_dir = photo_emb_dir
        self.sketch_emb_dir = sketch_emb_dir
        self.noise_emb_dir = noise_emb_dir
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
        return 2160962 if self.train else 390000

    def make_generator(self):
        dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        
        # get a list of photo paths that we will loop through
        train_photo_paths, test_photo_paths = self.train_test_split()
        photo_paths = train_photo_paths if self.train else test_photo_paths
        n_photos = len(photo_paths)
        # blacklist_paths is important to make sure test images do not 
        # bleed into training images
        blacklist_paths = test_photo_paths if self.train else train_photo_paths

        # categories that a photo could be
        categories = os.listdir(self.photo_emb_dir)
        n_categories = len(categories)

        # we don't care about class imbalance here or order of showing examples, 
        # so we should loop each photo and generate 1 of each of the following:
        # - 1) (photo, same_photo sketch)
        # - 2) (photo, same_class sketch)
        # - 3) (photo, diff_class sketch)
        # - 4) (photo, noise)

        for i in xrange(n_photos):  # loop through each photo
            photo_path = photo_paths[i]
            photo = volatile_load(photo_path, dtype)

            # find all matching sketches for this photo.
            sketch_paths = sample_sketch_from_photo_path(photo_path, self.sketch_emb_dir, return_all=True)
            noise_paths = sample_sketch_from_photo_path(photo_path, self.noise_emb_dir, return_all=True)
            photo_same_class_paths = sample_photo_from_photo_path(photo_path, self.photo_emb_dir, same_class=True,
                                                                  blacklist=blacklist_paths, return_all=True)
            n_sketches = len(sketch_paths)
            n_noises = len(noise_paths)
            n_photos_same_class = len(photo_same_class_paths)
            n_photos_diff_class = len(photo_diff_class_paths)

            # we loop through and find all the sketches that match with the current photo. 
            # We first yield all of the matching (p, s) pairs.
            for j in xrange(n_sketches):
                sketch_path = sketch_paths[j]
                sketch = volatile_load(sketch_path, dtype)
                yield (photo, sketch, photo_path, sketch_path, SAME_PHOTO_EX)

            # we do the same for noise.
            for j in xrange(n_noises):
                noise_path = noise_paths[j]
                sketch = volatile_load(noise_path, dtype)
                yield (photo, sketch, photo_path, noise_path, NOISE_EX)

            # we find all the photos of the class. we loop through 1 sketch for each other 
            # of those photos and yield (p, s, pairs)
            for j in xrange(n_photos_same_class):
                _photo_path = photo_same_class_paths[j]
                _sketch_path = sample_sketch_from_photo_path(_photo_path, self.sketch_emb_dir)
                sketch = volatile_load(_sketch_path, dtype)
                yield (photo, sketch, photo_path, _sketch_path, SAME_CLASS_EX)

            # for photos of a different category, we will sample 1 photo for 
            # each other category; and for each of those photos, we will sample 1 sketch.
            for j in xrange(n_categories):
                _photo_paths = glob(os.path.join(photo_emb_dir, categories[j], '*'))
                _photo_path = np.random.choice(photo_paths)
                _sketch_path = sample_sketch_from_photo_path(_photo_path, self.sketch_emb_dir)
                sketch = volatile_load(_sketch_path, dtype)
                yield (photo, sketch, photo_path, _sketch_path, DIFF_CLASS_EX)


def volatile_load(path, dtype):
    obj = np.load(path)[np.newaxis, :]
    obj = Variable(torch.from_numpy(obj), volatile=True).type(dtype)
    return obj
            

def sample_sketch_from_photo_path(photo_path, sketch_emb_dir, deterministic=False,
                                  return_all=False):
    """Find a corresponding sketch for each photo.

    :param photo_path: string path to photo
    :param sketch_emb_dir: path to folder of all sketches
    :param return_all: if True, return all match sketches, else sample 1
    :param deterministic: if True, return the 1st matching sketch, else sample 1
    """
    photo_name, photo_ext = os.path.splitext(os.path.basename(photo_path))
    photo_folder = os.path.dirname(photo_path).split('/')[-1]

    sketch_paths = glob(os.path.join(sketch_emb_dir, photo_folder, 
                        '{name}-*{ext}'.format(name=photo_name, ext=photo_ext)))
    if return_all:
        return sketch_paths

    sketch_path = sketch_paths[0] if deterministic else np.random.choice(sketch_paths)
    return sketch_path


def sample_photo_from_photo_path(photo_path, photo_emb_dir, blacklist=[], 
                                 same_class=True, return_all=False):
    """Get another photo that is related to the present photo is some way.

    :param blacklist: add photo_paths in here that you want to ignore 
                      i.e. that are from the testing set.
    :param same_class: whether the photo to return is from the same class 
                       as the input photo or not.
    :param return_all: if True, return all match sketches, else sample 1
    """
    photo_name, photo_ext = os.path.splitext(os.path.basename(photo_path))
    photo_folder = os.path.dirname(photo_path).split('/')[-1]

    if same_class:
        photo_paths = glob(os.path.join(photo_emb_dir, photo_folder, '*'))
        photo_paths = list(set(photo_paths) - set(blacklist))
        photo_paths.remove(photo_path)
    else:
        categories = os.listdir(photo_emb_dir)
        categories.remove(photo_folder)
        category = np.random.choice(categories)
        # no need to worry about blacklist here
        photo_paths = glob(os.path.join(photo_emb_dir, category, '*'))

    if return_all:
        return photo_paths

    return np.random.choice(photo_paths)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_dir', type=str)
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('--hard', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    Generator = HardGenerator if args.hard else EasyGenerator
    generator = Generator(args.photo_emb_dir, args.sketch_emb_dir,
                          batch_size=1, train=True,
                          use_cuda=args.cuda)
    examples = generator.make_generator()

    print("Testing generator of size %d." % generator.size)
    photos1, sketches1, labels1 = examples.next()
    photos2, sketches2, labels2 = examples.next()

    print('---------------')
    print('Photos batch 1:', photos1)
    print('Sketches batch 1:', sketches1)
    print('Labels batch 1:', labels1)
    print('---------------')
    print('Photos batch 2:', photos2)
    print('Sketches batch 2:', sketches2)
    print('Labels batch 2:', labels2)
