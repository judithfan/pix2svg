from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import csv
import shutil
import random
import numpy as np
from glob import glob

import torch
from torch.autograd import Variable


class ReferenceGenerator(object):
    def __init__(self, render_emb_dir, sketch_emb_dir, train=True,
                 batch_size=10, use_cuda=False):
        self.render_emb_dir = render_emb_dir
        self.sketch_emb_dir = sketch_emb_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.train = train

        with open('../reference_games/data/sketchpad_basic_merged_group_data.csv') as fp:
            csv_data = []
            reader = csv.reader(fp)
            for row in reader:
                csv_data.append(row)

        header = csv_data[0]
        csv_data = csv_data[1:]

        # all the important columns each csv row
        condition_ix = header.index('condition')
        gameid_ix = header.index('gameID')
        trialnum_ix = header.index('trialNum')
        target_ix = header.index('target')
        distractor1_ix = header.index('Distractor1')
        distractor2_ix = header.index('Distractor2')
        distractor3_ix = header.index('Distractor3')
        pose_ix = header.index('pose')

        target_lookup = {}
        distractor_lookup = {}
        count = 0

        for row in csv_data:
            if row[condition_ix] == 'closer':  # only care about close trials
                if row[pose_ix] != 'NA':
                    sketch_name = 'gameID_{id}_trial_{trial}.npy'.format(
                        id=row[gameid_ix], trial=row[trialnum_ix])
                    target_category = row[target_ix]
                    target_name = '{cat}_{id:04d}.npy'.format(
                        cat=row[target_ix], id=int(row[pose_ix]))
                    
                    distractor_names = []
                    # because we can't guarantee distractors to have been used as 
                    # the target in another trial, we need to increase the probability
                    # of that being the same. So we match a given target with distractors
                    # that are near by poses (within 18% each direction).
                    for i in xrange(-2, 3):
                        pose = min(max(int(row[pose_ix]) + i, 0), 39)
                        _distractor_names = [
                            '{cat}_{id:04d}.npy'.format(cat=row[distractor1_ix], id=pose), 
                            '{cat}_{id:04d}.npy'.format(cat=row[distractor2_ix], id=pose),
                            '{cat}_{id:04d}.npy'.format(cat=row[distractor3_ix], id=pose),
                        ]
                        distractor_names += _distractor_names
                    distractor_names = list(set(distractor_names))

                    if target_name in target_lookup:
                        target_lookup[target_name].append(sketch_name)
                    else:
                        target_lookup[target_name] = [sketch_name]
                    joint_name = '{target}+{sketch}'.format(target=target_name, sketch=sketch_name)
                    distractor_lookup[joint_name] = distractor_names
                    count += 1

        # target_lookup returns list of sketches given a particular render
        self.target_lookup = target_lookup
        # distractor lookup returns distractor renderings given render+sketch
        self.distractor_lookup = distractor_lookup
 
        self.cat_to_group = {
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

        train_paths, test_paths = self.train_test_split()
        self.size = len(train_paths) if self.train else len(test_paths)

    def train_test_split(self):
        render_paths = self.target_lookup.keys()
        render_cats = [self.cat_to_group[i.split('_')[0]] for i in render_paths]
        path_to_cat = dict(zip(render_paths, render_cats))

        train_paths = [i for i in render_paths if path_to_cat[i] != 'bird']
        test_paths = [i for i in render_paths if path_to_cat[i] == 'bird']
        random.shuffle(train_paths)
        random.shuffle(test_paths)

        return train_paths, test_paths

    def try_generator(self):
        train_paths, test_paths = self.train_test_split()
        render_paths = train_paths if self.train else test_paths
        
        while True:
            render1_path = random.choice(render_paths)
            sketch1_path = random.choice(self.target_lookup[render1_path])
            key = '{target}+{sketch}'.format(target=render1_path, sketch=sketch1_path)
            distractor_paths = self.distractor_lookup[key]
            distractor_paths = [path for path in distractor_paths 
                                if path in self.target_lookup]
            if len(distractor_paths) == 0:
                continue  # skip path
            
            render2_path = random.choice(distractor_paths)
            sketch2_path = random.choice(self.target_lookup[render2_path])
            break

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
            sketch1_path = random.choice(self.target_lookup[render1_path])
            # build key to lookup distractor renderings from render
            key = '{target}+{sketch}'.format(target=render1_path, sketch=sketch1_path)
            # this tells us the category of a distractor. here we care that the distractor
            # is an object of the same pose.
            distractor_paths = self.distractor_lookup[key]
            # only keep distractor paths where the path is in the target_lookup
            distractor_paths = [path for path in distractor_paths 
                                if path in self.target_lookup]
            if len(distractor_paths) == 0:
                continue  # skip path
            render2_path = random.choice(distractor_paths)
            # pick the sketch path that was by the same person as sketch1
            sketch2_path = random.choice(self.target_lookup[render2_path])

            # add full path
            render1_path = glob(os.path.join(self.render_emb_dir, '*' + render1_path))[0]
            sketch1_path = os.path.join(self.sketch_emb_dir, sketch1_path)
            render2_path = glob(os.path.join(self.render_emb_dir, '*' + render2_path))[0]
            sketch2_path = os.path.join(self.sketch_emb_dir, sketch2_path)

            # load paths into numpy
            render1 = np.load(render1_path)[np.newaxis, ...]
            sketch1 = np.load(sketch1_path)[np.newaxis, ...]
            render2 = np.load(render2_path)[np.newaxis, ...]
            sketch2 = np.load(sketch2_path)[np.newaxis, ...]
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='number of images to sample.')
    args = parser.parse_args()

    render_emb_dir = '/data/jefan/sketchpad_basic_extract/subordinate_allrotations_6_minified_conv_4_2'
    sketch_emb_dir = '/data/jefan/sketchpad_basic_extract/sketch_conv_4_2/'
    generator = ReferenceGenerator(render_emb_dir, sketch_emb_dir, train=True)

    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.makedirs('./tmp')

    for i in xrange(args.n):
        os.makedirs('./tmp/example%d' % i)

    emb_to_raw_path = lambda path: path.replace('_conv_4_2', '').replace('.npy', '.png')

    for i in xrange(args.n):
        files = generator.try_generator()
        render1 = emb_to_raw_path(os.path.join(render_emb_dir, '*' + files['render1']))
        sketch1 = emb_to_raw_path(os.path.join(sketch_emb_dir, files['sketch1']))
        render2 = emb_to_raw_path(os.path.join(render_emb_dir, '*' + files['render2']))
        sketch2 = emb_to_raw_path(os.path.join(sketch_emb_dir, files['sketch2']))
       
        render1 = glob(render1)[0]
        render2 = glob(render2)[0]
        
        shutil.copy(render1, './tmp/example%d/render1.png' % i)
        shutil.copy(sketch1, './tmp/example%d/sketch1.png' % i)
        shutil.copy(render2, './tmp/example%d/render2.png' % i)
        shutil.copy(sketch2, './tmp/example%d/sketch2.png' % i)
