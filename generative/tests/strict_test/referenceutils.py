from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import csv
import random
import numpy as np
from glob import glob


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

        for row in csv_data:
            if row[condition_ix] == 'closer':  # only care about close trials
                if row[pose_ix] != 'NA':
                    sketch_name = 'gameID_{id}_trial_{trial}.npy'.format(
                        id=row[gameid_ix], trial=row[trialnum_ix])
                    target_name = '{cat}_{id:04d}.npy'.format(
                        cat=row[target_ix], id=int(row[pose_ix]))
                    distractor_names = [
                        '{cat}_{id:04d}.npy'.format(cat=row[distractor1_ix], 
                                                    id=int(row[pose_ix])),
                        '{cat}_{id:04d}.npy'.format(cat=row[distractor2_ix], 
                                                    id=int(row[pose_ix])),
                        '{cat}_{id:04d}.npy'.format(cat=row[distractor3_ix], 
                                                    id=int(row[pose_ix])),
                    ]
                    if target_name in target_lookup:
                        target_lookup[target_name].append(sketch_name)
                    else:
                        target_lookup[target_name] = [sketch_name]

                    joint_name = '{target}+{sketch}'.format(target=target_name, sketch=sketch_name)
                    distractor_lookup[joint_name] = distractor_names
        
        # target_lookup returns list of sketches given photo
        self.target_lookup = target_lookup
        # distractor lookup returns distractor photo given photo+sketch
        self.distractor_lookup = distractor_lookup
        
        self.size = int(target_lookup.keys() * 0.8)
        if not train:
            self.size = len(target_lookup.keys()) - self.size
    
    def train_test_split(self):
        render_paths = self.target_lookup.keys()
        split = int(0.8 * len(render_paths))
        train_paths = render_paths[:split]
        test_paths = render_paths[split:]
        random.shuffle(train_paths)
        random.shuffle(test_paths)

        return train_photos, test_photos

    def make_generator(self):
        dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        train_paths, test_paths = self.train_test_split()
        render_paths = train_paths if self.train else test_paths
        blacklist_paths = test_paths if self.train else train_paths

        batch_idx = 0  # keep track of when to start new batch
        for i in range(self.size):
            # define (p1, s1), (p1, s2), (p2, s1), (p2, s2) paths
            render1_path = render_paths[i]
            sketch1_path = os.path.join(self.sketch_emb_dir, 
                                        random.choice(self.target_lookup[os.basename(render1_path)]))
            # build key to lookup distractor renderings from render
            key = '{target}+{sketch}'.format(target=os.path.basename(render1_path),
                                             sketch=os.path.basename(sketch1_path))
            render2_path = os.path.join(self.render_emb_dir, 
                                        random.choice(self.distractor_lookup[key]))
            # pick the sketch path that was by the same person as sketch1
            sketch2_paths = self.target_lookup[os.basename(render2_path)]
            sketch1_basepath = os.path.basename(sketch1_path)
            sketch1_gameid = sketch1_basepath.split('_')[1]
            r = re.compile("gameID_{}_trial_.*".format(sketch1_gameid))
            sketch2_path = filter(r.match, sketch2_paths)
            assert len(sketch2_path) == 1
            sketch2_path = os.path.join(self.sketch_emb_dir, sketch2_path[0])

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