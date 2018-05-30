from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import copy
import cPickle
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data.dataset import Dataset

OBJECT_TO_CATEGORY = {
    'basset': 'dog', 'beetle': 'car', 'bloodhound': 'dog', 'bluejay': 'bird',
    'bluesedan': 'car', 'bluesport': 'car', 'brown': 'car', 'bullmastiff': 'dog',
    'chihuahua': 'dog', 'crow': 'bird', 'cuckoo': 'bird', 'doberman': 'dog',
    'goldenretriever': 'dog', 'hatchback': 'car', 'inlay': 'chair', 'knob': 'chair',
    'leather': 'chair', 'nightingale': 'bird', 'pigeon': 'bird', 'pug': 'dog',
    'redantique': 'car', 'redsport': 'car', 'robin': 'bird', 'sling': 'chair',
    'sparrow': 'bird', 'squat': 'chair', 'straight': 'chair', 'tomtit': 'bird',
    'waiting': 'chair', 'weimaraner': 'dog', 'white': 'car', 'woven': 'chair',
}
CATEGORY_TO_OBJECT = {
    'dog': ['basset', 'bloodhound', 'bullmastiff', 'chihuahua', 'doberman', 'goldenretriever', 'pug', 'weimaraner'],
    'car': ['beetle', 'bluesedan', 'bluesport', 'brown', 'hatchback', 'redantique', 'redsport', 'white'],
    'bird': ['bluejay', 'crow', 'cuckoo', 'nightingale', 'pigeon', 'robin', 'sparrow', 'tomtit'],
    'chair': ['inlay', 'knob', 'leather', 'sling', 'squat', 'straight', 'waiting', 'woven'],
}

base_path = '/mnt/visual_communication_dataset/'
if os.uname()[1] == 'node8-neuroaicluster':
    base_path = '/data/jefan/'


class VisualDataset(Dataset):
    def __init__(self, layer='fc6', split='train', average_labels=False, 
                 overwrite_train_test_split=False, photo_transform=None, 
                 sketch_transform=None, train_test_split_dir='./train_test_split/1', 
                 random_seed=42):
        super(VisualDataset, self).__init__()
        np.random.seed(random_seed); random.seed(random_seed)
        db_path = base_path + 'sketchpad_basic_fixedpose96_%s' % layer
        sketch_dirname = os.path.join(db_path, 'sketch')
        photo_dirname = os.path.join(db_path, 'photos')
        sketch_basepaths = os.listdir(sketch_dirname)

        # load all the different things that we will use to remove paths
        valid_game_ids = pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))
        valid_game_ids = np.asarray(valid_game_ids['valid_gameids']).tolist()
        with open(os.path.join(db_path, 'invalid_trial_paths_pilot2.txt')) as fp:
            invalid_basepaths = [x.strip().replace('.png', '.npy') for x in fp.readlines()]
        with open(os.path.join(db_path, 'incorrect_trial_paths_pilot2.txt')) as fp:
            incorrect_basepaths = [x.strip().replace('.png', '.npy') for x in fp.readlines()]
            incorrect_basepaths = ['_'.join(x.split('_')[:-1]) + '.npy' for x in incorrect_basepaths]
        # do the actual removal of the paths
        sketch_basepaths = [path for path in sketch_basepaths 
                            if os.path.basename(path).split('_')[1] in valid_game_ids]
        sketch_basepaths = set(sketch_basepaths) - set(invalid_basepaths) - set(incorrect_basepaths)
        sketch_basepaths = list(sketch_basepaths)

        # this details how labels are stored (order of objects)
        object_order = pd.read_csv(base_path + 'human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()

        # load all 32 of them once since for every sketch we use the same 32 photos
        photo32_paths = [object_name + '.npy' for object_name in object_order]

        # load which sketches go to which classes
        with open(os.path.join(db_path, 'sketchpad_label_dict.pickle')) as fp:
            self.label_dict = cPickle.load(fp)

        with open(os.path.join(db_path, 'sketchpad_context_dict.pickle')) as fp:
            self.context_dict = cPickle.load(fp)

        # load human annotated labels.
        annotation_path = 'sketchpad_basic_recog_group_data_2_augmented.csv'
        annotations = pd.read_csv(os.path.join(base_path, annotation_path))
        annotations = zip(annotations['fname'].values, annotations['choice'].values)

        unrolled_dataset = defaultdict(lambda: [])
        for annotation, choice in annotations:
            annotation = annotation.replace('.png', '.npy')
            choice = object_order.index(choice)
            unrolled_dataset[annotation].append(choice)

        average_annotations = np.load(base_path+'human_confusion.npy')

        self.object_order = object_order
        preloaded_split = os.path.join(train_test_split_dir, '%s_split.json' % split)
        if os.path.isfile(preloaded_split) or overwrite_train_test_split:
            with open(preloaded_split) as fp:
                sketch_paths = json.load(fp)
        else:
            sketch_paths = self.train_test_split(split, sketch_basepaths, train_test_split_dir)

        if average_labels:
            sketch_dataset = []
            for path in sketch_paths:
                object_ = self.label_dict[path]
                object_ix = object_order.index(object_)
                context = self.context_dict[path]
                context_ix = 0 if context == 'closer' else 1
                labels = average_annotations[object_ix, :, context_ix]
                sketch_dataset.append((path, labels))
        else:
            sketch_dataset = []
            for path in sketch_paths:
                for label in unrolled_dataset[path]:
                    sketch_dataset.append((path, label))

        self.sketch_dataset = sketch_dataset
        self.sketch_dirname = sketch_dirname 
        self.photo_dirname = photo_dirname
        self.size = len(sketch_dataset)
        self.split = split
        self.object_order = object_order
        self.photo32_paths = photo32_paths
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform

    def train_test_split(self, split, basepaths, out_dir):
        train_basepaths, val_basepaths, test_basepaths, extra_basepaths = [], [], [], []
        object_names = self.object_order
        sketch_objects = np.asarray([self.label_dict[basepath] for basepath in basepaths])
        basepaths = np.asarray(basepaths)

        # for each class we want to make sure that we balance the 
        # classes in the training dataset
        context_dict = self.context_dict

        # store difference here
        context_diff = {}
        for object_name in object_names:
            object_basepaths = basepaths[sketch_objects == object_name]
            np.random.shuffle(object_basepaths)
            object_contexts = np.array([self.context_dict[path] for path in object_basepaths])
            num_closer = sum(object_contexts == 'closer')
            num_further = sum(object_contexts == 'further')
            context_diff[object_name] = num_further - num_closer
            
            # we always want to have balance
            num_examples = min(num_closer, num_further)            
            for context in ['closer', 'further']:
                context_basepaths = object_basepaths[object_contexts == context]
                train_ix = int(0.8 * num_examples)
                val_ix = int(0.9 * num_examples)
                test_ix = int(num_examples)
                train_basepaths += context_basepaths[:train_ix].tolist()
                val_basepaths += context_basepaths[train_ix:val_ix].tolist()
                test_basepaths += context_basepaths[val_ix:test_ix].tolist()
                extra_basepaths += context_basepaths[test_ix:].tolist()

        random.shuffle(train_basepaths)
        random.shuffle(val_basepaths)
        random.shuffle(test_basepaths)
        random.shuffle(extra_basepaths)

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, 'train_split.json'), 'wb') as fp:
            json.dump(train_basepaths, fp)
        with open(os.path.join(out_dir, 'val_split.json'), 'wb') as fp:
            json.dump(val_basepaths, fp)
        with open(os.path.join(out_dir,  'test_split.json'), 'wb') as fp:
            json.dump(test_basepaths, fp)
        with open(os.path.join(out_dir, 'extra_split.json'), 'wb') as fp:
            json.dump(extra_basepaths, fp)

        if split == 'train':
            paths = train_basepaths
        elif split == 'val':
            paths = val_basepaths
        elif split == 'test':
            paths = test_basepaths
        elif split == 'extra':
            paths = extra_basepaths
        else:
            raise Exception('split %s not recognized.' % split)
        return paths

    def gen_photos(self):
        def generator():
	    for photo_path in self.photo32_paths:
                photo_object = os.path.splitext(photo_path)[0]
                photo = np.load(os.path.join(self.photo_dirname, photo_path))
                photo = torch.from_numpy(photo)
                if self.photo_transform:
                    photo = self.photo_transform(photo)
                photo = photo.unsqueeze(0)
                yield photo 
        return generator

    def __getitem__(self, index):
        sketch_path, label = self.sketch_dataset[index] 
        sketch = torch.from_numpy(np.load(os.path.join(self.sketch_dirname, sketch_path)))
        if self.sketch_transform:
            sketch = self.transform(sketch)
        return sketch, label

    def __len__(self):
        return self.size


class ExhaustiveDataset(VisualDataset):
    """Used to create the RDM and JSON. Loops through every sketch & photo pair."""
    def __init__(self, layer='fc6', split='test', photo_transform=None, sketch_transform=None, 
                 train_test_split_dir='./train_test_split/1', random_seed=42):
        super(ExhaustiveDataset, self).__init__()
        np.random.seed(random_seed); random.seed(random_seed)
        db_path = base_path + 'sketchpad_basic_fixedpose96_%s' % layer
        photo_dirname = os.path.join(db_path, 'photos')
        sketch_dirname = os.path.join(db_path, 'sketch')
        sketch_basepaths = os.listdir(sketch_dirname)

        # load all the different things that we will use to remove paths
        valid_game_ids = pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))
        valid_game_ids = np.asarray(valid_game_ids['valid_gameids']).tolist()
        with open(os.path.join(db_path, 'invalid_trial_paths_pilot2.txt')) as fp:
            invalid_basepaths = [x.strip().replace('.png', '.npy') for x in fp.readlines()]
        with open(os.path.join(db_path, 'incorrect_trial_paths_pilot2.txt')) as fp:
            incorrect_basepaths = [x.strip().replace('.png', '.npy') for x in fp.readlines()]
            incorrect_basepaths = ['_'.join(x.split('_')[:-1]) + '.npy' for x in incorrect_basepaths]
        # do the actual removal of the paths
        sketch_basepaths = [path for path in sketch_basepaths 
                            if os.path.basename(path).split('_')[1] in valid_game_ids]
        sketch_basepaths = set(sketch_basepaths) - set(invalid_basepaths) - set(incorrect_basepaths)
        sketch_basepaths = list(sketch_basepaths)

        # this details how labels are stored (order of objects)
        object_order = pd.read_csv(base_path+'human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()
        
        # load all 32 of them once since for every sketch we use the same 32 photos
        photo_32_paths = [object_name + '.npy' for object_name in object_order]
        
        with open(os.path.join(db_path, 'sketchpad_context_dict.pickle')) as fp:
            self.context_dict = cPickle.load(fp)
        
        with open(os.path.join(db_path, 'sketchpad_label_dict.pickle')) as fp:
            self.label_dict = cPickle.load(fp)

        if split != 'full':
            preloaded_split = os.path.join(train_test_split_dir, '%s_split.json' % split)
            if os.path.isfile(preloaded_split):
                with open(preloaded_split) as fp:
                    sketch_basepaths = json.load(fp)
            else:
                sketch_basepaths = self.train_test_split(split, sketch_basepaths)

        sketch_paths = [os.path.join(sketch_dirname, path) for path in sketch_basepaths]
        self.sketch_dirname = sketch_dirname 
        self.photo_dirname = photo_dirname
        self.size = len(sketch_paths)
        self.sketch_paths = sketch_paths
        self.photo_32_paths = photo_32_paths
        self.object_order = object_order
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform

    def gen_photos(self):
        def generator():
            photo_paths = self.photo_32_paths
            for photo_path in photo_paths:
                photo_object = os.path.splitext(photo_path)[0]
                photo = np.load(os.path.join(self.photo_dirname, photo_path))
                photo = torch.from_numpy(photo)
                if self.photo_transform:
                    photo = self.photo_transform(photo)
                photo = photo.unsqueeze(0)
                yield photo, photo_object, photo_path
        return generator

    def __getitem__(self, index):
        sketch_path = self.sketch_paths[index]
        context = self.context_dict[os.path.basename(sketch_path)]
        sketch_object = self.label_dict[os.path.basename(sketch_path)]
        sketch = np.load(sketch_path)
        sketch = torch.from_numpy(sketch)
        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)
        return sketch, sketch_object, context, sketch_path

    def __len__(self):
        return self.size
