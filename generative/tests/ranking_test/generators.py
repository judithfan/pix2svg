"""We will try to explicitly encode some structure in our model
by showing tuples of (photo, sketch [same photo], sketch [same class], 
sketch [different class], noise) and enforce a ranking loss.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random
import numpy as np
from glob import glob

import torch
from torch.autograd import Variable


class RankingGenerator(object):
    """This data generator returns 
    (photo, sketch_same_photo, sketch_same_class, sketch_diff_class, noise).
    Note that this generator only accepts embeddings (not raw images).
 
    :param photo_emb_dir: path to photo embedding directories
    :param sketch_emb_dir: path to sketch embedding directories
    :param noise_emb_dir: path to noise embedding directories
    :param train: if True, return 80% of data; else return 20%.
    :param batch_size: number to return at a time
    :param use_cuda: if True, make CUDA compatible objects
    :param strict: if True, sketches of the same class but different photo are
                   treated as negatives. The good part of doing this is to really
                   pull apart exact photo sketch matches. The bad part is that
                   noise and diff photo same class are about the same. Ideally, 
                   we want to have noise + diff class be about the same, and have
                   same class diff photo and same class same photo near each other.
    """
    def __init__(self, photo_emb_dir, sketch_emb_dir, noise_emb_dir, batch_size=32, 
                 train=True, strict=False, use_cuda=False):
        self.photo_emb_dir = photo_emb_dir
        self.sketch_emb_dir = sketch_emb_dir
        self.noise_emb_dir = noise_emb_dir
        self.batch_size = batch_size
        self.train = train
        self.strict = strict
        self.size = self.get_size()

    def get_size(self):
        categories = os.listdir(self.sketch_emb_dir)
        n_categories = len(categories)
        categories = (categories[:int(n_categories * 0.8)] if self.train
                      else categories[int(n_categories * 0.8):])
        sketch_paths = [path for path in list_files(self.sketch_emb_dir, ext='npy')
                        if os.path.dirname(path).split('/')[-1] in categories]
        return len(sketch_paths)

    def make_generator(self):
        # automatically handle cuda calls
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        # sketches and photos are organized into classes by directories. we want
        # to take advantage of this fact to do train/test splits.
        categories = os.listdir(self.sketch_emb_dir)
        n_categories = len(categories)
        # depending on whether this is train or test, we will return a different 
        # subset of photos/sketches/noise to work with; that way we don't have any
        # questions of generalizability.
        categories = (categories[:int(n_categories * 0.8)] if self.train 
                      else categories[int(n_categories * 0.8):])
        photo_paths = [path for path in list_files(self.photo_emb_dir, ext='npy') 
                       if os.path.dirname(path).split('/')[-1] in categories]
        sketch_paths = [path for path in list_files(self.sketch_emb_dir, ext='npy') 
                       if os.path.dirname(path).split('/')[-1] in categories]
        noise_paths = [path for path in list_files(self.noise_emb_dir, ext='npy')
                       if os.path.dirname(path).split('/')[-1] in categories]

        # everytime we call this, the order will be different -- a sort of implicit
        # regularization.
        random.shuffle(sketch_paths)
        batch_idx = 0

        # for each sketch, find its matching photo, find a sketch of same class but different photo, 
        # find a sketch of different class, and find the matching noise. The generator will go through
        # each sketch like this. It is not guaranteed that all sketch eswill be used for same-class/diff-photo
        # or for diff-class.
        for i in range(len(sketch_paths)):
            sketch_same_photo_path = sketch_paths[i]
            photo_path = get_photo_from_sketch_path(sketch_same_photo_path, self.photo_emb_dir)
            noise_path = get_noise_from_sketch_path(sketch_same_photo_path, self.noise_emb_dir)
            # note that these 2 functions are random processes. therefore, repeated calls will show 
            # different pairs (still in dataset); this is desired as another form of data 
            # augmentation / implicit regularization.
            sketch_same_class_path = get_same_class_sketch_from_photo(photo_path, self.sketch_emb_dir)
            sketch_diff_class_path = get_diff_class_sketch_from_photo(photo_path, self.sketch_emb_dir, 
                                                                      categories)

            # consider all 5 images at once.
            photo = np.load(photo_path)
            sketch_same_photo = np.load(sketch_same_photo_path)
            sketch_same_class = np.load(sketch_same_class_path)
            sketch_diff_class = np.load(sketch_diff_class_path)
            noise = np.load(noise_path)

            if batch_idx == 0:
                photo_batch = photo
                sketch_same_photo_batch = sketch_same_photo
                sketch_same_class_batch = sketch_same_class
                sketch_diff_class_batch = sketch_diff_class
                noise_batch = noise
            else:
                photo_batch = np.vstack((photo_batch, photo))
                sketch_same_photo_batch = np.vstack((sketch_same_photo_batch, sketch_same_photo))
                sketch_same_class_batch = np.vstack((sketch_same_class_batch, sketch_same_class))
                sketch_diff_class_batch = np.vstack((sketch_diff_class_batch, sketch_diff_class))
                noise_batch = np.vstack((noise_batch, noise))

            if (batch_idx + 1) == self.batch_size:
                photo_batch = Variable(torch.from_numpy(photo_batch), 
                                       volatile=not self.train).type(dtype)
                sketch_same_photo_batch = Variable(torch.from_numpy(sketch_same_photo_batch), 
                                                   volatile=not self.train).type(dtype)
                sketch_same_class_batch = Variable(torch.from_numpy(sketch_same_class_batch), 
                                                   volatile=not self.train).type(dtype)
                sketch_diff_class_batch = Variable(torch.from_numpy(sketch_diff_class_batch), 
                                                   volatile=not self.train).type(dtype)
                noise_batch = Variable(torch.from_numpy(noise_batch), 
                                       volatile=not self.train).type(dtype)

                yield (photo_batch, sketch_same_photo_batch, sketch_same_class_batch,
                       sketch_diff_class_batch, noise_batch)
                
                batch_idx = -1
            # increment batch_idx
            batch_idx += 1

        # return any remaining data
        if batch_idx > 0:
            photo_batch = Variable(torch.from_numpy(photo_batch), 
                                   volatile=not self.train).type(dtype)
            noise_batch = Variable(torch.from_numpy(noise_batch), 
                                   volatile=not self.train).type(dtype)
            sketch_same_photo_batch = Variable(torch.from_numpy(sketch_same_photo_batch), 
                                               volatile=not self.train).type(dtype)
            sketch_same_class_batch = Variable(torch.from_numpy(sketch_same_class_batch), 
                                               volatile=not self.train).type(dtype)
            sketch_diff_class_batch = Variable(torch.from_numpy(sketch_diff_class_batch), 
                                               volatile=not self.train).type(dtype)

            yield (photo_batch, sketch_same_photo_batch, sketch_same_class_batch,
                   sketch_diff_class_batch, noise_batch)


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
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result


if __name__ == "__main__":
    import time
    # test our generator -- esp. with timings because I'm worried about it
    # being a little bit slow:
    photo_emb_dir = '/home/wumike/full_sketchy_embeddings/photos'
    sketch_emb_dir = '/home/wumike/full_sketchy_embeddings/sketches'
    noise_emb_dir = '/home/wumike/full_sketchy_embeddings/noise'

    generator = EmbeddingGenerator(photo_emb_dir, sketch_emb_dir,
                                   noise_emb_dir, 32, train=True)
    generator = generator.make_generator()

    start_time = time.time()
    for i in xrange(100):
        print('Iteration [{}/{}]'.format(i + 1, 100))
        generator.next()

    print('\nWall Time: {} seconds'.format(time.time() - start_time))
