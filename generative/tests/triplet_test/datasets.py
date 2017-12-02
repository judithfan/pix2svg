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
import pandas as pd
from glob import glob
from collections import defaultdict
from itertools import combinations

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

CATEGORY_TO_INSTANCE_DICT = {
    'dog': ['basset', 'bloodhound', 'bullmastiff', 'chihuahua', 'doberman', 'goldenretriever', 'pug', 'weimaraner'],
    'car': ['beetle', 'bluesedan', 'bluesport', 'brown', 'hatchback', 'redantique', 'redsport', 'white'],
    'bird': ['bluejay', 'crow', 'cuckoo', 'nightingale', 'pigeon', 'robin', 'sparrow', 'tomtit'],
    'chair': ['inlay', 'knob', 'leather', 'sling', 'squat', 'straight', 'waiting', 'woven'],
}

INSTANCE_IX2NAME_DICT = {0: 'basset', 1: 'beetle', 2: 'bloodhound', 3: 'bluejay', 4: 'bluesedan',
                         5: 'bluesport', 6: 'brown', 7: 'bullmastiff', 8: 'chihuahua', 9: 'crow',
                         10: 'cuckoo', 11: 'doberman', 12: 'goldenretriever', 13: 'hatchback', 14: 'inlay',
                         15: 'knob', 16: 'leather', 17: 'nightingale', 18: 'pigeon', 19: 'pug',
                         20: 'redantique', 21: 'redsport', 22: 'robin', 23: 'sling', 24: 'sparrow',
                         25: 'squat', 26: 'straight', 27: 'tomtit', 28: 'waiting', 29: 'weimaraner',
                         30: 'white', 31: 'woven'}
INSTANCE_NAME2IX_DICT = {v: k for k, v in INSTANCE_IX2NAME_DICT.iteritems()}
CATEGORY_IX2NAME_DICT = {0: 'bird', 1: 'car', 2: 'chair', 3: 'dog'}
CATEGORY_NAME2IX_DICT = {v: k for k, v in CATEGORY_IX2NAME_DICT.iteritems()}



class Generator(object):
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

        # only game ids in here are allowed
        good_games = pd.read_csv(os.path.join(self.data_dir, 'valid_gameids_pilot2.csv'))['valid_gameids']
        good_games = good_games.values.tolist()

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
        # map from target to condition
        target2condition = {}

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

            # we ignore sketches that were bad games
            if _row_gameid not in good_games:
                continue

            # dog/car/chair/bird
            target_category = row[target_ix]
            target_name = '{sketch}_{category}.npy'.format(sketch=sketch_base,
                                                           category=target_category)
            target2condition[target_name] = row[condition_ix]
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
        self.target2condition = target2condition

        train_paths, test_paths = self.train_test_split()
        self.size = len(train_paths) if self.train else len(test_paths)

    def train_test_split(self):
        raise NotImplementedError

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
            # convert to torch
            render_group = torch.from_numpy(render_group).type(dtype)
            sketch_group = torch.from_numpy(sketch_group).type(dtype)
            label_group = torch.from_numpy(label_group).type(dtype)
            # gen category metrics
            sketch1_category = CATEGORY_NAME2IX_DICT[gen_category_from_path(render1_path)]
            sketch2_category = CATEGORY_NAME2IX_DICT[gen_category_from_path(render2_path)]
            sketch_categories = torch.Tensor([sketch1_category, sketch2_category, sketch1_category, sketch2_category]).type(dtype).long()
            # gen instance metrics
            sketch1_instance = INSTANCE_NAME2IX_DICT[gen_instance_from_path(render1_path)]
            sketch2_instance = INSTANCE_NAME2IX_DICT[gen_instance_from_path(render2_path)]
            sketch_instances = torch.Tensor([sketch1_instance, sketch2_instance, sketch1_instance, sketch2_instance]).type(dtype).long()

            if batch_idx == 0:
                render_batch = render_group
                sketch_batch = sketch_group
                label_batch = label_group
                sketch_cat_batch = sketch_categories
                sketch_inst_batch = sketch_instances
            else:
                render_batch = torch.cat((render_batch, render_group))
                sketch_batch = torch.cat((sketch_batch, sketch_group))
                label_batch = torch.cat((label_batch, label_group))
                sketch_cat_batch = torch.cat((sketch_cat_batch, sketch_categories))
                sketch_inst_batch = torch.cat((sketch_inst_batch, sketch_instances))

            batch_idx += 1

            if batch_idx == self.batch_size:
                render_batch = Variable(render_batch)
                sketch_batch = Variable(sketch_batch)
                label_batch = Variable(label_batch, requires_grad=False)
                sketch_cat_batch = Variable(sketch_cat_batch, requires_grad=False)
                sketch_inst_batch = Variable(sketch_inst_batch, requires_grad=False)
                
                yield (render_batch, sketch_batch, label_batch, 
                       sketch_cat_batch, sketch_inst_batch)
                batch_idx = 0

        if batch_idx > 0:
            render_batch = Variable(render_batch)
            sketch_batch = Variable(sketch_batch)
            label_batch = Variable(label_batch, requires_grad=False)
            sketch_cat_batch = Variable(sketch_cat_batch, requires_grad=False)
            sketch_inst_batch = Variable(sketch_inst_batch, requires_grad=False)

            yield (render_batch, sketch_batch, label_batch,
                   sketch_cat_batch, sketch_inst_batch)


class PreloadedGenerator(Generator):
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
        self.target2condition = data['target2condition']

        train_paths, test_paths = self.train_test_split()
        self.size = len(train_paths) if train else len(test_paths)

    def gen_pickle_name(self):
        raise NotImplementedError


class ContextFreeGenerator(Generator):
    """Splits randomly but makes sure that training and test do not share
    any of the same contexts."""
    def gen_unique_contexts(self):
        contexts = []
        for k, v in self.target2distractors.iteritems():
            k = os.path.splitext(k)[0].split('_')[-1]
            v = [os.path.splitext(i)[0].split('_')[-1] for i in v]
            context = [k] + v
            context = sorted(context)
            contexts.append(context)
        unique_contexts = [list(x) for x in 
                           set(tuple(x) for x in contexts)]
        return unique_contexts

    def train_test_split(self):
        random.seed(42)
        np.random.seed(42)

        contexts = self.gen_unique_contexts()
        random.shuffle(contexts)
        n_contexts = len(contexts)
        n_train = int(n_contexts * 0.80)  # 80/20 train/test split

        train_contexts = set(tuple(x) for x in contexts[:n_train])
        test_contexts = set(tuple(x) for x in contexts[n_train:])

        all_paths = []
        for paths in self.cat2target.itervalues():
            all_paths += paths
        random.shuffle(all_paths)

        train_paths, test_paths = [], []
        for path in all_paths:
            target = path
            distractors = self.target2distractors[target]
            target = os.path.splitext(target)[0].split('_')[-1]
            distractors = [os.path.splitext(i)[0].split('_')[-1] for i in distractors]
            context = [target] + distractors
            context = sorted(context)

            assert not (tuple(context) in train_contexts and \
                tuple(context) in test_contexts)

            if tuple(context) in train_contexts:
                train_paths.append(path)
            elif tuple(context) in test_contexts:
                test_paths.append(path)
            else:
                raise Exception('How did you get here?')

        return train_paths, test_paths


class ContextFreePreloadedGenerator(PreloadedGenerator):
    def gen_unique_contexts(self):
        contexts = []
        for k, v in self.target2distractors.iteritems():
            k = os.path.splitext(k)[0].split('_')[-1]
            v = [os.path.splitext(i)[0].split('_')[-1] for i in v]
            context = [k] + v
            context = sorted(context)
            contexts.append(context)
        unique_contexts = [list(x) for x in 
                           set(tuple(x) for x in contexts)]
        return unique_contexts

    def train_test_split(self):
        random.seed(42)
        np.random.seed(42)

        contexts = self.gen_unique_contexts()
        random.shuffle(contexts)
        n_contexts = len(contexts)
        n_train = int(n_contexts * 0.60)  # 60/40 train/test split

        train_contexts = set(tuple(x) for x in contexts[:n_train])
        test_contexts = set(tuple(x) for x in contexts[n_train:])

        all_paths = []
        for paths in self.cat2target.itervalues():
            all_paths += paths
        random.shuffle(all_paths)

        train_paths, test_paths = [], []
        for path in all_paths:
            target = path
            distractors = self.target2distractors[target]
            target = os.path.splitext(target)[0].split('_')[-1]
            distractors = [os.path.splitext(i)[0].split('_')[-1] for i in distractors]
            context = [target] + distractors
            context = sorted(context)

            assert not (tuple(context) in train_contexts and \
                tuple(context) in test_contexts)

            if tuple(context) in train_contexts:
                train_paths.append(path)
            elif tuple(context) in test_contexts:
                test_paths.append(path)
            else:
                raise Exception('How did you get here?')

        return train_paths, test_paths

    def gen_pickle_name(self):
        return ('preloaded_context_closer.pkl'
                if self.closer_only else 'preloaded_context_all.pkl')


def gen_category_from_path(path):
    instance = gen_instance_from_path(path)
    return CATEGORY_LOOKUP[instance]


def gen_instance_from_path(path):
    return os.path.splitext(path)[0].split('_')[-1]


if __name__ == "__main__":
    """Running this will generate the pickle file to run PreloadedGenerator"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=str, help='conv_4_2|fc7', default='conv_4_2')
    parser.add_argument('--photo_augment', action='store_true')
    args = parser.parse_args()
    assert args.layer in ['conv_4_2', 'fc7']
    
    if args.photo_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose96_photo_augmented_%s' % args.layer
    else:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose96_%s' % args.layer

    generator = ContextFreeGenerator(data_dir=data_dir)
    with open(os.path.join(data_dir, 'preloaded_context_all.pkl'), 'wb') as fp:
        cPickle.dump({'cat2target': generator.cat2target, 
                      'target2sketch': generator.target2sketch,
                      'distractor2sketch': generator.distractor2sketch,
                      'target2distractors': generator.target2distractors,
                      'path2folder': generator.path2folder,
                      'target2condition': generator.target2condition}, fp)
    
    print('Pickle saved.')
