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


class MultiModalTrainGenerator(object):
    """This data generator returns (photo, sketch, label) where label
    is 0 or 1: same object, different object.
 
    :param photo_emb_dir: pass to photo embedding directories
    :param sketch_emb_dir: pass to sketch embedding directories
    :param batch_size: number to return at a time
    :param use_cuda: if True, make CUDA compatible objects
    :param strict: if True, sketches of the same class but different photo are
                   treated as negatives. The good part of doing this is to really
                   pull apart exact photo sketch matches. The bad part is that
                   noise and diff photo same class are about the same. Ideally, 
                   we want to have noise + diff class be about the same, and have
                   same class diff photo and same class same photo near each other.
    """
    def __init__(self, photo_emb_dir, sketch_emb_dir,
                 batch_size=32, use_cuda=False, strict=False):
        self.photo_emb_dir = photo_emb_dir
        self.sketch_emb_dir = sketch_emb_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.strict = strict
        self.size = self.get_size()

    def get_size(self):
        categories = os.listdir(self.sketch_emb_dir)
        n_categories = len(categories)
        categories = categories[:int(n_categories * 0.8)]
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
        categories = categories[:int(n_categories * 0.8)]
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

        # if strict, we treat same_class pairs as negatives, meaning we want to 
        # only treat same_photo as positive. This theoretically should preserve
        # more finer grain details in our embedding.
        if self.strict: 
            same_photo_ixs = [SAME_PHOTO_EX for i in range(n_paths)]
            same_class_ixs = [SAME_CLASS_EX for i in range(n_paths / 2)]
            diff_class_ixs = [DIFF_CLASS_EX for i in range(n_paths / 2)]
            sample_ixs = same_photo_ixs + same_class_ixs + diff_class_ixs
            random.shuffle(sample_ixs)
        else:  # otherwise, we just treat diff_class as negatives and let the 
            # model fill in the blanks.
            same_photo_ixs = [SAME_PHOTO_EX for i in range(n_paths)]
            diff_class_ixs = [DIFF_CLASS_EX for i in range(n_paths)]
            sample_ixs = same_photo_ixs + diff_class_ixs
            random.shuffle(sample_ixs)

        # None is how we keep track of batches.
        photo_batch = None 
        sketch_batch = None 
        label_batch = None
        type_batch = None

        for i in range(n_paths):
            sketch_path = sketch_paths[i]
            if sample_ixs[i] == SAME_PHOTO_EX:
                photo_path = get_photo_from_sketch_path(
                    sketch_path, self.photo_emb_dir)
                label = 1
            elif sample_ixs[i] == SAME_CLASS_EX:
                photo_path = get_same_class_photo_from_sketch(
                    sketch_path, self.photo_emb_dir)
                label = 0 if self.strict else 1
            elif sample_ixs[i] == DIFF_CLASS_EX:
                photo_path = get_diff_class_photo_from_sketch(
                    sketch_path, self.photo_emb_dir, categories)
                label = 0
            else:
                raise Exception('Example type %d not recognized.' % sample_ixs[i])

            photo = np.load(photo_path)
            sketch = np.load(sketch_path)

            if photo_batch is None:
                photo_batch = photo[np.newaxis, ...]
                sketch_batch = sketch[np.newaxis, ...]
                label_batch = [label]
                type_batch = [sample_ixs[i]]
            else:
                photo_batch = np.vstack((photo_batch, photo))
                sketch_batch = np.vstack((sketch_batch, sketch))
                label_batch.append(label)
                type_batch.append(sample_ixs[i])

            if photo_batch.shape[0] == self.batch_size:
                # this is for training, so volatile is False
                photo_batch = Variable(torch.from_numpy(photo_batch)).type(dtype)
                sketch_batch = Variable(torch.from_numpy(sketch_batch)).type(dtype)
                label_batch = Variable(torch.from_numpy(np.array(label_batch)), 
                                       requires_grad=False).type(dtype)
                type_batch = Variable(torch.from_numpy(np.array(type_batch)), 
                                      requires_grad=False).type(dtype)
                
                # yield data so we can continue to query the same object
                yield (photo_batch, sketch_batch, label_batch, type_batch)
                photo_batch = None
                sketch_batch = None 
                label_batch = None
                type_batch = None

        # return any remaining data
        # since n_data may not be divisible by batch_size
        if photo_batch is not None:
            photo_batch = Variable(torch.from_numpy(photo_batch)).type(dtype)
            sketch_batch = Variable(torch.from_numpy(sketch_batch)).type(dtype)
            label_batch = Variable(torch.from_numpy(np.array(label_batch)), 
                                   requires_grad=False).type(dtype)
            type_batch = Variable(torch.from_numpy(np.array(type_batch)), 
                                  requires_grad=False).type(dtype)
 
            yield (photo_batch, sketch_batch, label_batch, type_batch)
            photo_batch = None
            sketch_batch = None 
            label_batch = None
            type_batch = None


class MultiModalTestGenerator(object):
    """This data generator returns (photo, sketch, label) where label
    is 0 or 1: same object, different object. This is very similar to 
    the MultiModalTrainGenerator but it has no class imbalance 
    constraints and also includes a noise directory.

    :param photo_emb_dir: pass to photo embedding directories
    :param sketch_emb_dir: pass to sketch embedding directories
    :param noise_emb_dir: if not None, mix in noise samples in negatives.
    :param batch_size: number to return at a time
    :param use_cuda: if True, make CUDA compatible objects
    :param strict: if True, sketches of the same class but different photo are
                   treated as negatives. The good part of doing this is to really
                   pull apart exact photo sketch matches. The bad part is that
                   noise and diff photo same class are about the same. Ideally, 
                   we want to have noise + diff class be about the same, and have
                   same class diff photo and same class same photo near each other.
    """
    def __init__(self, photo_emb_dir, sketch_emb_dir, noise_emb_dir=None,
                 batch_size=32, strict=False, use_cuda=False):
        self.photo_emb_dir = photo_emb_dir
        self.sketch_emb_dir = sketch_emb_dir
        self.noise_emb_dir = noise_emb_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.strict = strict
        self.size = self.get_size()

    def get_size(self):
        categories = os.listdir(self.sketch_emb_dir)
        n_categories = len(categories)
        categories = categories[int(n_categories * 0.8):]
        sketch_paths = [path for path in list_files(self.sketch_emb_dir, ext='npy') 
                        if os.path.dirname(path).split('/')[-1] in categories]
        return len(sketch_paths) * 4

    def make_generator(self):
        dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # note that everytime this is called, it will return slightly different
        # pairs since not all pairs are used in 1 epoch. But it will never
        # intersect with the train directory.
        categories = os.listdir(self.sketch_emb_dir)
        n_categories = len(categories)

        # test is composed of sketches from different categories. This 
        # clearly distinguishes which sketches and photos are being used
        # to train and which are for testing.
        categories = categories[int(n_categories * 0.8):]
        photo_paths = [path for path in list_files(self.photo_emb_dir, ext='npy') 
                       if os.path.dirname(path).split('/')[-1] in categories]
        sketch_paths = [path for path in list_files(self.sketch_emb_dir, ext='npy') 
                       if os.path.dirname(path).split('/')[-1] in categories]
        if self.noise_emb_dir:
            noise_paths = [path for path in list_files(self.noise_emb_dir, ext='npy') 
                           if os.path.dirname(path).split('/')[-1] in categories]

        # we don't care about class imbalance here, so we should loop through 
        # all sketches four times, once per kind of pair:
        # - 1) (photo, same_photo sketch)
        # - 2) (photo, same_class sketch)
        # - 3) (photo, diff_class sketch)
        # - 4) (photo, noise)
        n_paths = len(sketch_paths) * (4 if self.noise_emb_dir else 3)
        
        same_photo_ixs = [SAME_PHOTO_EX for i in range(n_paths)]
        same_class_ixs = [SAME_CLASS_EX for i in range(n_paths)]
        diff_class_ixs = [DIFF_CLASS_EX for i in range(n_paths)]
        sample_ixs = same_photo_ixs + same_class_ixs + diff_class_ixs 

        if self.noise_emb_dir:
            noise_ixs = [NOISE_EX for i in range(n_paths)]
            sample_ixs += noise_ixs
        random.shuffle(sample_ixs)

        # None is how we keep track of batches.
        photo_batch = None
        sketch_batch = None 
        label_batch = None
        type_batch = None

        for i in range(n_paths):
            sketch_path = sketch_paths[i]
            if sample_ixs[i] == SAME_PHOTO_EX:
                photo_path = get_photo_from_sketch_path(
                    sketch_path, self.photo_emb_dir)
                label = 1
            elif sample_ixs[i] == SAME_CLASS_EX:
                photo_path = get_same_class_photo_from_sketch(
                    sketch_path, self.photo_emb_dir)
                label = 0 if self.strict else 1
            elif sample_ixs[i] == DIFF_CLASS_EX:
                photo_path = get_diff_class_photo_from_sketch(
                    sketch_path, self.photo_emb_dir, categories)
                label = 0
            elif sample_ixs[i] == NOISE_EX:
                photo_path = get_photo_from_sketch_path(
                    sketch_path, self.photo_emb_dir)
                # replace sketch here with a sketch made from
                # random noise.
                sketch_path = get_noise_from_sketch_path(
                    sketch_paths, self.noise_emb_dir)
                label = 0
            else:
                raise Exception('Example type %d not recognized.' % sample_ixs[i])

            photo = np.load(photo_path)
            sketch = np.load(sketch_path)

            if photo_batch is None:
                photo_batch = photo
                sketch_batch = sketch
                label_batch = [label]
                type_batch = [sample_ixs[i]]
            else:
                photo_batch = np.vstack((photo_batch, photo))
                sketch_batch = np.vstack((sketch_batch, sketch))
                label_batch.append(label)
                type_batch.append(sample_ixs[i])

            if photo_batch.shape[0] == self.batch_size:
                # this is for training, so volatile is False
                photo_batch = Variable(torch.from_numpy(photo_batch)).type(dtype)
                sketch_batch = Variable(torch.from_numpy(sketch_batch)).type(dtype)
                label_batch = Variable(torch.from_numpy(np.array(label_batch)), 
                                       requires_grad=False).type(dtype)
                type_batch = Variable(torch.from_numpy(np.array(type_batch)), 
                                      requires_grad=False).type(dtype)
                
                # yield data so we can continue to query the same object
                yield (photo_batch, sketch_batch, label_batch, type_batch)
                photo_batch = None
                sketch_batch = None 
                label_batch = None
                type_batch = None

        # return any remaining data
        # since n_data may not be divisible by batch_size
        if photo_batch is not None:
            photo_batch = Variable(torch.from_numpy(photo_batch)).type(dtype)
            sketch_batch = Variable(torch.from_numpy(sketch_batch)).type(dtype)
            label_batch = Variable(torch.from_numpy(np.array(label_batch)), 
                                   requires_grad=False).type(dtype)
            type_batch = Variable(torch.from_numpy(np.array(type_batch)), 
                                  requires_grad=False).type(dtype)
 
            yield (photo_batch, sketch_batch, label_batch, type_batch)
            photo_batch = None
            sketch_batch = None 
            label_batch = None
            type_batch = None


# 100000 loops, best of 3: 2.37 micros per loop
def get_photo_from_sketch_path(sketch_path, photo_emb_dir):
    """Get path to matching photo given sketch path"""
    sketch_filename = os.path.basename(sketch_path)
    sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
    photo_filename = sketch_filename.split('-')[0] + '.npy'
    photo_path = os.path.join(photo_emb_dir, sketch_folder, photo_filename)
    return photo_path


# 100000 loops, best of 3: 2.21 micros per loop
def get_noise_from_sketch_path(sketch_path, noise_emb_dir):
    sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
    sketch_filename = os.path.basename(sketch_path)
    noise_path = os.path.join(noise_emb_dir, sketch_folder, sketch_filename)
    return noise_path


# 1000 loops, best of 3: 2.07 ms per loop
def get_same_class_photo_from_sketch(sketch_path, photo_emb_dir):
    """Find a photo of the same class as the sketch but not 
    of the same sketch.
    
    :param sketch_path: path to sketch embedding.
    :param photo_emb_dir: directory pointing to photo embeddings
    """
    sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
    # only keep photo paths of the same class
    photo_paths = list_files(os.path.join(photo_emb_dir, sketch_folder), ext='npy')
    # remove the photo of our sketch from consideration
    blacklist_path = '{}{}'.format(os.path.basename(sketch_path).split('-')[0],
                                   os.path.splitext(os.path.basename(sketch_path))[1])
    photo_paths = list(set(photo_paths) - set([blacklist_path]))
    photo_path = np.random.choice(photo_paths)
    return photo_path


# 100 loops, best of 3: 2.36 ms per loop
def get_diff_class_photo_from_sketch(sketch_path, photo_emb_dir, categories):
    """Find a sketch of a different class.

    :param sketch_path: path to sketch embedding.
    :param photo_emb_dir: directory pointing to photo embeddings
    :param categories: list of classes
    """
    sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
    # pick a random category that is not our sketches' category
    category = np.random.choice(list(set(categories) - set([sketch_folder])))
    photo_paths = list_files(os.path.join(photo_emb_dir, category), ext='npy')
    photo_path = np.random.choice(photo_paths)
    return photo_path


# 1000 loops, best of 3: 2.07 ms per loop
def get_same_class_sketch_from_photo(photo_path, sketch_emb_dir):
    """Find a sketch of the same class as the photo but not of the same photo.
    
    :param photo_path: path to photo embedding.
    :param sketch_paths: list of paths to all sketches.
    """
    photo_folder = os.path.dirname(photo_path).split('/')[-1]
    photo_filename = os.path.basename(photo_path)
    photo_filename = os.path.splitext(photo_filename)[0]

    # only keep sketch paths of the same class
    sketch_paths = list_files(os.path.join(sketch_emb_dir, photo_folder), ext='npy')
    # there are never more than 20 sketches for a given photo. this 
    # is hacky but to avoid loops, we can do set subtraction by assuming
    # that there 20 sketches for this sketch
    blacklist_paths = [os.path.join(sketch_emb_dir, photo_folder, 
                                   '{}-{}.npy'.format(photo_filename, i))
                       for i in range(20)]
    sketch_paths = list(set(sketch_paths) - set(blacklist_paths))
    sketch_path = np.random.choice(sketch_paths)
    return sketch_path


# 100 loops, best of 3: 2.36 ms per loop
def get_diff_class_sketch_from_photo(photo_path, sketch_emb_dir, categories):
    """Find a sketch of a different class.

    :param photo_path: path to photo embedding.
    :param sketch_emb_dir: directory pointing to sketch embeddings
    :param categories: list of classes
    """
    photo_folder = os.path.dirname(photo_path).split('/')[-1]
    category = np.random.choice(list(set(categories) - set([photo_folder])))
    sketch_paths = list_files(os.path.join(sketch_emb_dir, category), ext='npy')
    sketch_path = np.random.choice(sketch_paths)
    return sketch_path


def list_files(path, ext='jpg'):
    # recursively lise files in folder and all subfolders
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n_examples', type=int, help='number of calls to make')
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    photo_emb_dir = '/data/wumike/full_sketchy_embeddings_fc7/photos'
    sketch_emb_dir = '/data/wumike/full_sketchy_embeddings_fc7/sketches'
    noise_emb_dir = '/data/wumike/full_sketchy_embeddings_fc7/noise'

    if args.test:
        generator = MultiModalTestGenerator(photo_emb_dir, sketch_emb_dir, 
                                            noise_emb_dir=noise_emb_dir, batch_size=1)
    else:
        generator = MultiModalTrainGenerator(photo_emb_dir, sketch_emb_dir, batch_size=1)
    examples = generator.make_generator()

    photo_lst = []
    sketch_lst = []
    label_lst = []
    type_lst = []

    for i in range(args.n_examples):
        photos, sketchs, labels, types = examples.next()
        photo_lst.append(photos)
        sketch_lst.append(sketchs)
        label_lst.append(labels)
        type_lst.append(types)

    photo_lst = torch.cat(photo_lst)
    sketch_lst = torch.cat(sketch_lst)
    label_lst = torch.cat(label_lst)
    type_lst = torch.cat(type_lst)

    print('\nPhotos:')
    print(photo_lst)
    print('\nSketches:')
    print(sketch_lst)
    print('\nLabels:')
    print(label_lst)
    print('\nTypes:')
    print(type_lst)
