from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import cPickle
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data.dataset import Dataset
 

class SketchPlusPhotoDataset(Dataset):
    def __init__(self, layer='fc6', split='train', soft_labels=False, photo_transform=None, 
                 sketch_transform=None, random_seed=42):
        super(Dataset, self).__init__()
        np.random.seed(random_seed)
        random.seed(random_seed)
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % layer
        photos_dirname = os.path.join(db_path, 'photos')
        sketch_dirname = os.path.join(db_path, 'sketch')
        sketch_basepaths = os.listdir(sketch_dirname)

        valid_game_ids = np.asarray(pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))['valid_gameids']).tolist()
        # only keep sketches that are in the valid_gameids (some games are garbage)
        sketch_basepaths = [path for path in sketch_basepaths if os.path.basename(path).split('_')[1] in valid_game_ids]

        # this details how labels are stored (order of objects)
        object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()
        
        # load all 32 of them once since for every sketch we use the same 32 photos
        photo_32_paths = [os.path.join(photos_dirname, object_name + '.npy') for object_name in object_order]

        # load human annotated labels.
        self.annotations = np.load('/mnt/visual_communication_dataset/human_confusion.npy')

        # load which sketches go to which contexts
        with open(os.path.join(db_path, 'sketchpad_context_dict.pickle')) as fp:
            self.context_dict = cPickle.load(fp)

        # load which sketches go to which classes
        with open(os.path.join(db_path, 'sketchpad_label_dict.pickle')) as fp:
            self.label_dict = cPickle.load(fp)
            # reverse label from string to integer
            self.reverse_label_dict = defaultdict(lambda: [])
            for path, label in self.label_dict.iteritems():
                self.reverse_label_dict[label].append(path)

        sketch_paths = self.train_test_split(split, sketch_dirname, sketch_basepaths)

        self.split = split
        self.use_soft_labels = soft_labels
        self.photo_32_paths = photo_32_paths
        self.object_order = object_order
        self.sketch_paths = sketch_paths
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform
        
        # negative sample + add labels
        self.preprocess_data()

    def train_test_split(self, split, sketch_dirname, sketch_basepaths):
        # find which trials to use in training
        trial_nums = [int(os.path.splitext(path)[0].split('_')[-1]) for path in sketch_basepaths]
        uniq_trial_nums = list(set(trial_nums))
        train_trial_nums = uniq_trial_nums[:int(0.8 * len(uniq_trial_nums))]
        test_trial_nums = uniq_trial_nums[int(0.8 * len(uniq_trial_nums)):]
        train_sketch_basepaths = [path for path in sketch_basepaths 
                                  if int(os.path.splitext(path)[0].split('_')[-1]) in train_trial_nums]
        test_sketch_basepaths = [path for path in sketch_basepaths 
                                 if int(os.path.splitext(path)[0].split('_')[-1]) in test_trial_nums]

        # make sure closer and further are balanced
        all_closer_basepaths = [path for path in sketch_basepaths if self.context_dict[path] == 'closer']
        all_further_basepaths = [path for path in sketch_basepaths if self.context_dict[path] == 'further']

        train_closer_basepaths = list(set(all_closer_basepaths).intersection(set(train_sketch_basepaths)))
        train_further_basepaths = list(set(all_further_basepaths).intersection(set(train_sketch_basepaths)))
        n_closer_train = len(train_closer_basepaths)
        n_further_train = len(train_further_basepaths)

        if n_closer_train > n_further_train:
            n_diff = n_closer_train - n_further_train
            train_sketch_basepaths = list(set(train_sketch_basepaths) - set(train_closer_basepaths[:n_diff]))
        elif n_further_train > n_closer_train:
            n_diff = n_further_train - n_closer_train
            train_sketch_basepaths = list(set(train_sketch_basepaths) - set(train_further_basepaths[:n_diff]))

        _train_sketch_paths = [os.path.join(sketch_dirname, path) for path in train_sketch_basepaths]
        random.shuffle(_train_sketch_paths)
        train_sketch_paths = _train_sketch_paths[:int(0.8 * len(_train_sketch_paths))]
        val_sketch_paths = _train_sketch_paths[int(0.8 * len(_train_sketch_paths)):]
        test_sketch_paths = [os.path.join(sketch_dirname, path) for path in test_sketch_basepaths]

        if split == 'train':
            sketch_paths = train_sketch_paths
        elif split == 'val':
            sketch_paths = val_sketch_paths
        else:  # split == 'test'
            sketch_paths = test_sketch_paths
        return sketch_paths

    def preprocess_data(self):
        pos_photo_paths, pos_sketch_paths, neg_photo_paths, neg_sketch_paths = \
            self.negative_sample(self.photo_32_paths, self.sketch_paths)
        photo_paths = pos_photo_paths + neg_photo_paths
        sketch_paths = pos_sketch_paths + neg_sketch_paths
        hard_labels = ([1 for _ in xrange(len(pos_sketch_paths))] + 
                       [0 for _ in xrange(len(neg_sketch_paths))])
        soft_labels = [self.annotations[self.object_order.index(self.label_dict[os.path.basename(sketch)]), 
                                        self.object_order.index(os.path.splitext(os.path.basename(photo))[0]),
                                        0 if self.context_dict[os.path.basename(sketch)] == 'closer' else 1] 
                       for photo, sketch in zip(photo_paths, sketch_paths)]
        examples = zip(photo_paths, sketch_paths, hard_labels, soft_labels)
        random.shuffle(examples)
        photo_paths, sketch_paths, hard_labels, soft_labels = zip(*examples)
        self.photo_data = photo_paths
        self.sketch_data = sketch_paths
        self.hard_labels = hard_labels
        self.soft_labels = soft_labels
        self.size = len(sketch_paths)

    def negative_sample(self, photo_paths, sketch_paths):
        neg_sketch_paths = []
        neg_photo_paths = []
        pos_sketch_paths = []
        pos_photo_paths = []
        
        print('Building negative samples.')
        for i in tqdm(xrange(len(sketch_paths))):
            sketch_path = sketch_paths[i]
            object1 = self.label_dict[os.path.basename(sketch_path)]
            object1_ix = self.object_order.index(object1)
            object2 = random.choice(list(set(self.object_order) - set([object1])))
            object2_ix = self.object_order.index(object2)
            neg_sketch_path = random.choice(self.reverse_label_dict[object2])
            neg_sketch_path = os.path.join(os.path.dirname(sketch_path), neg_sketch_path)
            neg_sketch_paths.append(neg_sketch_path)
            neg_photo_paths.append(photo_paths[object2_ix])
            pos_sketch_paths.append(sketch_path)
            pos_photo_paths.append(photo_paths[object1_ix])

        return pos_photo_paths, pos_sketch_paths, neg_photo_paths, neg_sketch_paths

    def __getitem__(self, index):
        photo = torch.from_numpy(np.load(self.photo_data[index]))
        sketch = torch.from_numpy(np.load(self.sketch_data[index]))
        label = self.soft_labels[index] if self.use_soft_labels else self.hard_labels[index]

        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)

        if self.photo_transform:
            photo = self.photo_transform(photo)

        return photo, sketch, label

    def __len__(self):
        return self.size


class ObjectSplitDataset(SketchPlusPhotoDataset):
    def train_test_split(self, split, sketch_dirname, sketch_basepaths):
        train_sketch_basepaths = []
        val_sketch_basepaths = []
        test_sketch_basepaths = []

        object_names = self.reverse_label_dict.keys()
        sketch_objects = np.asarray([self.label_dict[basepath] for basepath in sketch_basepaths])
        sketch_basepaths = np.asarray(sketch_basepaths)

        for object_name in object_names:
            object_basepaths = sketch_basepaths[sketch_objects == object_name].tolist()
            num_basepaths = len(object_basepaths)
            num_train = int(0.8 * num_basepaths)
            num_val = int(0.9 * num_basepaths) - num_train
            random.shuffle(object_basepaths)
            train_sketch_basepaths += object_basepaths[:num_train]
            val_sketch_basepaths += object_basepaths[num_train:num_train+num_val]
            test_sketch_basepaths += object_basepaths[num_train+num_val:]

        train_sketch_paths = [os.path.join(sketch_dirname, path) for path in train_sketch_basepaths]
        val_sketch_paths = [os.path.join(sketch_dirname, path) for path in val_sketch_basepaths]
        test_sketch_paths = [os.path.join(sketch_dirname, path) for path in test_sketch_basepaths]

        if split == 'train':
            sketch_paths = train_sketch_paths
        elif split == 'val':
            sketch_paths = val_sketch_paths
        else:  # split == 'test'
            sketch_paths = test_sketch_paths
        return sketch_paths     


class TargetedGeneralization(SketchPlusPhotoDataset):
    def __init__(self, test_sketch_basepath, layer='fc6', split='train', soft_labels=False, photo_transform=None,
                 sketch_transform=None, random_seed=42):
        self.test_sketch_basepath = test_sketch_basepath
        super(TargetedGeneralization, self).__init__(layer=layer, split=split, soft_labels=soft_labels, photo_transform=photo_transform,
                                                     sketch_transform=sketch_transform, random_seed=random_seed)

    def train_test_split(self, split, sketch_dirname, sketch_basepaths):
        test_sketch_basepaths = [self.test_sketch_basepath]
        other_sketch_basepaths = list(set(sketch_basepaths) - set(test_sketch_basepaths))
        random.shuffle(other_sketch_basepaths)
        train_sketch_basepaths = other_sketch_basepaths[:int(0.8 * len(other_sketch_basepaths))]
        val_sketch_basepaths = other_sketch_basepaths[int(0.8 * len(other_sketch_basepaths)):]

        train_sketch_paths = [os.path.join(sketch_dirname, path) for path in train_sketch_basepaths]
        val_sketch_paths = [os.path.join(sketch_dirname, path) for path in val_sketch_basepaths]
        test_sketch_paths = [os.path.join(sketch_dirname, path) for path in test_sketch_basepaths]

        if split == 'train':
            sketch_paths = train_sketch_paths
        elif split == 'val':
            sketch_paths = val_sketch_paths
        else:  # split == 'test'
            sketch_paths = test_sketch_paths
        return sketch_paths

