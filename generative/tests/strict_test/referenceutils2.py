"""Clone of referenceutils.py but for the new data which is organized
in a more sane way where we do not need to worry about cache misses.
Also, no more pose! so we don't need to worry about that variability.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import csv
import shutil
import random
import cPickle
import numpy as np
from glob import glob
from collections import defaultdict

import torch
from torch.autograd import Variable


CATEGORY_LOOKUP = {
    'basset': 'dog',
    'beetle': 'car',
    'bloodhound': 'dog',
    'bluejay': 'bird',
    'bluesedan': 'car',
    'bluesport': 'car',
    'brown': 'car',
    'bullmastiff': 'dog',
    'chihuahua': 'dog',
    'crow': 'bird',
    'cuckoo': 'bird',
    'doberman': 'dog',
    'goldenretriever': 'dog',
    'hatchback': 'car',
    'inlay': 'chair',
    'knob': 'chair',
    'leather': 'chair',
    'nightingale': 'bird',
    'pigeon': 'bird',
    'pug': 'dog',
    'redantique': 'car',
    'redsport': 'car',
    'robin': 'bird',
    'sling': 'chair',
    'sparrow': 'bird',
    'squat': 'chair',
    'straight': 'chair',
    'tomtit': 'bird',
    'waiting': 'chair',
    'weimaraner': 'dog',
    'white': 'car',
    'woven': 'chair',
}


class ThreeClassGenerator(object):
    """This takes all in images in 3 classes and uses them as 
    training; it keeps the last class for testing. This is meant
    to measure cross-class generalization in sketchpad.

    :param train: whether to yield training or testing examples
    :param batch_size: number of examples to return at once
    :param use_cuda: whether to use CUDA objects or not
    """

    def __init__(self, train=True, batch_size=10, use_cuda=False, closer_only=False,
                 data_dir='/data/jefan/sketchpad_basic_fixedpose_conv_4_2'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.train = train
        self.closer_only = closer_only

        with open(os.path.join(self.data_dir, 'incorrect_trial_paths_pilot2.txt')) as fp:
            # these games need to be ignored
            bad_games = fp.readlines()

        with open(os.path.join(self.data_dir, 'sketchpad_basic_pilot2_group_data.csv')) as fp:
            csv_data = []
            reader = csv.reader(fp)
            for row in reader:
                csv_data.append(row)
            # open csv data and load data
            header = csv_data[0]
            csv_data = csv_data[1:]

        # map from category to sketch path
        cat2target = {'car': [], 'dog': [], 'chair': [], 'bird': []}
        # map from target to sketch
        target2sketch = {}
        # map from target to distractors
        target2distractors = {}
        # map from distractor to sketch
        distractor2sketch = {}
        # map target/distractor/sketch to folder
        path2folder = {}

        # indexes for different data items that might be useful
        condition_ix = header.index('condition')
        gameid_ix = header.index('gameID')
        trialnum_ix = header.index('trialNum')
        target_ix = header.index('target')
        distractors_ix = [header.index('Distractor1'),
                          header.index('Distractor2'),
                          header.index('Distractor3')]

        for ir, row in enumerate(csv_data):
            print('Initializing row [{}/{}]'.format(ir + 1, len(csv_data)))
            # we only care about closer cases right now
            if closer_only and row[condition_ix] != 'closer':
                continue

            row_gameid = row[gameid_ix]
            row_trialnum = row[trialnum_ix]

            # _row_gameid = row_gamid sans crop
            _row_gameid = row_gameid
            is_crop = '-crop' in row_gameid
            if is_crop:
                ii = row_gameid.index('-crop')
                _row_gameid = row_gameid[:ii]

            sketch_base = 'gameID_{id}_trial_{trial}'.format(
                id=row_gameid, trial=row_trialnum)
            sketch_name = '{sketch}.npy'.format(sketch=sketch_base)

            # we ignore sketches that were in bad games
            if sketch_name.replace('.npy', '.png\n') in bad_games:
                continue

            # dog/car/chair/bird
            target_category = row[target_ix]
            target_name = '{sketch}_{category}.npy'.format(sketch=sketch_base,
                                                           category=target_category)
            target2distractors[target_name] = []

            path2folder[sketch_name] = os.path.join(self.data_dir, 'sketch')
            path2folder[target_name] = os.path.join(self.data_dir, 'target')

            for k, ix in enumerate(distractors_ix):
                distractor_category = row[ix]
                distractor_name = '{sketch}_{category}.npy'.format(sketch=sketch_base, 
                                                                   category=distractor_category)
                if is_crop:
                    # if the current sketch is a crop, then find a distractor sketch which is
                    # also a crop; we can't guarantee same crop location
                    distractor_target_regex = 'gameID_{id}-crop*_trial_*_{category}.npy'.format(
                        id=_row_gameid, category=distractor_category)
                    matches = glob(os.path.join(self.data_dir, 'target', distractor_target_regex))
                    distractor_target_name = random.choice(matches)
                else:
                    distractor_target_regex = 'gameID_{id}_trial_*_{category}.npy'.format(
                        id=_row_gameid, category=distractor_category)
                    matches = glob(os.path.join(self.data_dir, 'target', distractor_target_regex))
                    assert len(matches) == 1
                    distractor_target_name = matches[0]

                distractor_target_name = os.path.basename(distractor_target_name)
                # find sketch for matching target
                distractor_sketch_name = '_'.join(distractor_target_name.split('_')[:-1]) + '.npy'
                # ssave of all of this information so we can look it up later
                distractor2sketch[distractor_name] = distractor_sketch_name
                target2distractors[target_name].append(distractor_name)
                path2folder[distractor_name] = os.path.join(self.data_dir, 'distractor%d' % (k+1))
                path2folder[distractor_sketch_name] = os.path.join(self.data_dir, 'sketch')

            cat2target[CATEGORY_LOOKUP[target_category]].append(target_name)
            target2sketch[target_name] = sketch_name

        self.cat2target = cat2target
        self.target2sketch = target2sketch
        self.distractor2sketch = distractor2sketch
        self.target2distractors = target2distractors
        self.path2folder = path2folder

        train_paths, test_paths = self.train_test_split()
        self.size = len(train_paths) if self.train else len(test_paths)

    def train_test_split(self):
        cat2target = self.cat2target
        train_paths = cat2target['car'] + cat2target['dog'] + cat2target['chair']
        test_paths = cat2target['bird']

        random.shuffle(train_paths)
        random.shuffle(test_paths)

        return train_paths, test_paths

    def try_generator(self):
        train_paths, test_paths = self.train_test_split()
        render_paths = train_paths if self.train else test_paths

        render1_path = random.choice(render_paths)
        sketch1_path = self.target2sketch[render1_path]
        render2_path = random.choice(self.target2distractors[render1_path])
        sketch2_path = self.distractor2sketch[render2_path]

        return {
            'render1': render1_path,
            'sketch1': sketch1_path,
            'render2': render2_path,
            'sketch2': sketch2_path,
        }

    def make_generator(self):
        dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        train_paths, test_paths = self.train_test_split()
        render_paths = train_paths if self.train else test_paths

        batch_idx = 0  # keep track of when to start new batch
        for i in range(self.size):
            # define (p1, s1), (p1, s2), (p2, s1), (p2, s2) paths
            render1_path = render_paths[i]
            sketch1_path = self.target2sketch[render1_path]
            render2_path = random.choice(self.target2distractors[render1_path])
            sketch2_path = self.distractor2sketch[render2_path]
            # render folders
            render1_dir = self.path2folder[render1_path]
            sketch1_dir = self.path2folder[sketch1_path]
            render2_dir = self.path2folder[render2_path]
            sketch2_dir = self.path2folder[sketch2_path]
            # load paths into numpy
            render1 = np.load(os.path.join(render1_dir, render1_path))[np.newaxis, ...]
            sketch1 = np.load(os.path.join(sketch1_dir, sketch1_path))[np.newaxis, ...]
            render2 = np.load(os.path.join(render2_dir, render2_path))[np.newaxis, ...]
            sketch2 = np.load(os.path.join(sketch2_dir, sketch2_path))[np.newaxis, ...]
            # organize into 4 pairs
            render_group = np.vstack((render1, render2, render1, render2))
            sketch_group = np.vstack((sketch1, sketch2, sketch2, sketch1))
            label_group = np.array((1, 1, 0, 0))

            if batch_idx == 0:
                render_batch = render_group
                sketch_batch = sketch_group
                label_batch = label_group
            else:
                render_batch = np.vstack((render_batch, render_group))
                sketch_batch = np.vstack((sketch_batch, sketch_group))
                label_batch = np.concatenate((label_batch, label_group))

            batch_idx += 1

            if batch_idx == self.batch_size:
                render_batch = Variable(torch.from_numpy(render_batch)).type(dtype)
                sketch_batch = Variable(torch.from_numpy(sketch_batch)).type(dtype)
                label_batch = Variable(torch.from_numpy(label_batch), 
                                       requires_grad=False).type(dtype)
                
                yield (render_batch, sketch_batch, label_batch)
                batch_idx = 0

        if batch_idx > 0:
            render_batch = Variable(torch.from_numpy(render_batch)).type(dtype)
            sketch_batch = Variable(torch.from_numpy(sketch_batch)).type(dtype)
            label_batch = Variable(torch.from_numpy(label_batch), 
                                   requires_grad=False).type(dtype)

            yield (render_batch, sketch_batch, label_batch)


class ThreeClassPreloadedGenerator(ThreeClassGenerator):
    """ThreeClassGenerator takes a long to build the dictionaries. 
    Preload them."""

    def __init__(self, train=True, batch_size=10, use_cuda=False, closer_only=False,
                 data_dir='/data/jefan/sketchpad_basic_fixedpose_conv_4_2'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.train = train
        self.closer_only = closer_only

        pickle_name = self.gen_pickle_name()
        with open(os.path.join(self.data_dir, pickle_name), 'r') as fp:
            data = cPickle.load(fp)

        self.cat2target = data['cat2target']
        self.target2sketch = data['target2sketch']
        self.distractor2sketch = data['distractor2sketch']
        self.target2distractors = data['target2distractors']
        self.path2folder = data['path2folder']

        train_paths, test_paths = self.train_test_split()
        self.size = len(train_paths) if train else len(test_paths)

    def gen_pickle_name(self):
        return ('preloaded_cross_closer.pkl' 
                if self.closer_only else 'preloaded_cross_all.pkl')


class FourClassGenerator(ThreeClassGenerator):
    """This takes the majority of images in 4 classes and uses them as 
    training while keeping the minority for testing. The way we 
    choose the majority is by taking images in 6/8 subclasses of each 
    class i.e. sparrow of bird. This is meant to measure intra-class 
    generalization in sketchpad.
    """
    
    def train_test_split(self):
        cat2target = self.cat2target
        train_paths, test_paths = [], []
        for paths in cat2target.itervalues():
            n = len(paths)
            n_train = int((n * 0.80) // 1)
            train_paths += paths[:n_train]
            test_paths += paths[n_train:]

        random.shuffle(train_paths)
        random.shuffle(test_paths)

        return train_paths, test_paths


class FourClassPreloadedGenerator(ThreeClassPreloadedGenerator):

    def train_test_split(self):
        cat2target = self.cat2target
        train_paths, test_paths = [], []
        for paths in cat2target.itervalues():
            n = len(paths)
            n_train = int((n * 0.80) // 1)
            train_paths += paths[:n_train]
            test_paths += paths[n_train:]

        random.shuffle(train_paths)
        random.shuffle(test_paths)

        return train_paths, test_paths

    def gen_pickle_name(self):
        return ('preloaded_intra_closer.pkl' 
                if self.closer_only else 'preloaded_intra_all.pkl')


class EntityGenerator(ThreeClassGenerator):
    """This splits by gameID so we can make claims about generalizing
    to unseen individuals."""
    
    def train_test_split(self):
        random.seed(42)
        np.random.seed(42)
        cat2target = self.cat2target
        all_paths = []
        all_games = []
        for paths in cat2target.itervalues():
            all_paths += paths
            all_games += [path.split('_')[1] for path in paths]
        # make this a unique set
        all_games = list(set(all_games))
        random.shuffle(all_games)
        n_games = len(all_games)
        n_train = int((n_games * 0.80) // 1)
        # split into training games and testing games
        train_games = all_games[:n_train]
        test_games = all_games[n_train:]
        # now we need to find actual paths
        train_paths, test_paths = [], []
        for path in all_paths:
            if path.split('_')[1] in train_games:
                train_paths.append(path)
            elif path.split('_')[1] in test_games:
                test_paths.append(path)
            else:
                raise Exception('Missing path: %s.' % path)
        return train_paths, test_paths


class EntityPreloadedGenerator(ThreeClassPreloadedGenerator):

    def train_test_split(self):
        random.seed(42)
        np.random.seed(42)
        cat2target = self.cat2target
        all_paths = []
        all_games = []
        for paths in cat2target.itervalues():
            all_paths += paths
            all_games += [path.split('_')[1] for path in paths]
        # make this a unique set
        all_games = list(set(all_games))
        random.shuffle(all_games)
        n_games = len(all_games)
        n_train = int((n_games * 0.80) // 1)
        # split into training games and testing games
        train_games = all_games[:n_train]
        test_games = all_games[n_train:]
        # now we need to find actual paths
        train_paths, test_paths = [], []
        for path in all_paths:
            if path.split('_')[1] in train_games:
                train_paths.append(path)
            elif path.split('_')[1] in test_games:
                test_paths.append(path)
            else:
                raise Exception('Missing path: %s.' % path)
        return train_paths, test_paths

    def gen_pickle_name(self):
        return ('preloaded_entity_closer.pkl' 
                if self.closer_only else 'preloaded_entity_all.pkl')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='number of images to sample.')
    parser.add_argument('generator', type=str, help='cross|intra|entity')
    parser.add_argument('--model', type=str, help='conv_4_2|fc7', default='conv_4_2')
    parser.add_argument('--closer', action='store_true', help='if True, include only closer examples')
    parser.add_argument('--photo_augment', action='store_true')
    parser.add_argument('--sketch_augment', action='store_true')
    parser.add_argument('--test', action='store_true', help='if True, sample from test set')
    args = parser.parse_args()
    args.train = not args.test

    assert args.generator in ['cross', 'intra', 'entity']
    assert args.model in ['conv_4_2', 'fc7']
    
    if args.photo_augment and args.sketch_augment:
        raise Exception('Cannot pass both photo_augment and sketch_augment')
    if args.photo_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose_augmented2_%s' % args.model
    elif args.sketch_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose_augmented_%s' % args.model
    else:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose_%s' % args.model

    if args.generator == 'cross':
        generator = ThreeClassPreloadedGenerator(train=args.train, batch_size=1, 
                                                 closer_only=args.closer, data_dir=data_dir)
    elif args.generator == 'intra':
        generator = FourClassPreloadedGenerator(train=args.train, batch_size=1, 
                                                closer_only=args.closer, data_dir=data_dir)
    elif args.generator == 'entity':
        generator = EntityPreloadedGenerator(train=args.train, batch_size=1, 
                                             closer_only=args.closer, data_dir=data_dir)

    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.makedirs('./tmp')

    for i in xrange(args.n):
        os.makedirs('./tmp/example%d' % i)

    emb_to_raw_path = lambda path: path.replace('.npy', '.png')
    render_dir = '/data/jefan/sketchpad_basic_fixedpose%s/*' % key
    sketch_dir = '/data/jefan/sketchpad_basic_fixedpose%s/sketch' % key

    for i in xrange(args.n):
        files = generator.try_generator()
        render1 = emb_to_raw_path(os.path.join(render_dir, files['render1']))
        sketch1 = emb_to_raw_path(os.path.join(sketch_dir, files['sketch1']))
        render2 = emb_to_raw_path(os.path.join(render_dir, files['render2']))
        sketch2 = emb_to_raw_path(os.path.join(sketch_dir, files['sketch2']))

        render1 = glob(render1)[0]
        render2 = glob(render2)[0]
        
        shutil.copy(render1, './tmp/example%d/render1.png' % i)
        shutil.copy(sketch1, './tmp/example%d/sketch1.png' % i)
        shutil.copy(render2, './tmp/example%d/render2.png' % i)
        shutil.copy(sketch2, './tmp/example%d/sketch2.png' % i)
