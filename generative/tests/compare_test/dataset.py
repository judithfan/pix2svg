from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
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
 

class VisualDataset(Dataset):
    def __init__(self, layer='fc6', split='train', photo_transform=None, sketch_transform=None, random_seed=42):
        super(VisualDataset, self).__init__()
        np.random.seed(random_seed); random.seed(random_seed)
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % layer
        sketch_dirname = os.path.join(db_path, 'sketch')
        photo_dirname = os.path.join(db_path, 'photos')
        sketch_basepaths = os.listdir(sketch_dirname)

        valid_game_ids = pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))
        valid_game_ids = np.asarray(valid_game_ids['valid_gameids']).tolist()
        sketch_basepaths = [path for path in sketch_basepaths 
                            if os.path.basename(path).split('_')[1] in valid_game_ids]

        # this details how labels are stored (order of objects)
        object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()

        # load all 32 of them once since for every sketch we use the same 32 photos
        photo32_paths = [object_name + '.npy' for object_name in object_order]

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

        sketch_paths = self.train_test_split(split, sketch_basepaths)
        self.sketch_dirname = sketch_dirname 
        self.photo_dirname = photo_dirname
        self.size = len(sketch_paths)
        self.split = split
        self.object_order = object_order
        self.photo32_paths = photo32_paths
        self.sketch_paths = sketch_paths
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform

    def train_test_split(self, split, basepaths):
        train_basepaths, val_basepaths, test_basepaths = [], [], []

        object_names = self.reverse_label_dict.keys()
        sketch_objects = np.asarray([self.label_dict[basepath] for basepath in basepaths])
        basepaths = np.asarray(basepaths)

        for object_name in object_names:
            object_basepaths = basepaths[sketch_objects == object_name].tolist()
            num_basepaths = len(object_basepaths)
            num_train = int(0.7 * num_basepaths)
            num_val = int(0.8 * num_basepaths) - num_train
            random.shuffle(object_basepaths)
            train_basepaths += object_basepaths[:num_train]
            val_basepaths += object_basepaths[num_train:num_train+num_val]
            test_basepaths += object_basepaths[num_train+num_val:]

        if split == 'train':
            paths = train_basepaths
        elif split == 'val':
            paths = val_basepaths
        else:  # split == 'test'
            paths = test_basepaths
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
        sketch_path = self.sketch_paths[index] 
        sketch_obj = self.label_dict[os.path.basename(sketch_path)]
        sketch_obj_ix = self.object_order.index(sketch_obj)
        sketch_category = OBJECT_TO_CATEGORY[sketch_obj]
        context = self.context_dict[os.path.basename(sketch_path)]
        context = 0 if context == 'closer' else 1
        sketch = torch.from_numpy(np.load(os.path.join(self.sketch_dirname, sketch_path)))

        if self.sketch_transform:
            sketch = self.transform(sketch)

        label = self.annotations[sketch_obj_ix, :, context]
        return sketch, label

    def __len__(self):
        return self.size


class SketchOnlyDataset(Dataset):
    def __init__(self, layer='fc6', split='train', transform=None, random_seed=42):
        super(Dataset, self).__init__()
        np.random.seed(random_seed); random.seed(random_seed)
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % layer
        dirname = os.path.join(db_path, 'sketch')
        basepaths = os.listdir(dirname)

        valid_game_ids = pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))
        valid_game_ids = np.asarray(valid_game_ids['valid_gameids']).tolist()
        basepaths = [path for path in basepaths if os.path.basename(path).split('_')[1] in valid_game_ids]

        # this details how labels are stored (order of objects)
        object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()

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

        paths = self.train_test_split(split, basepaths)
        self.dirname = dirname 
        self.size = len(paths)
        self.split = split
        self.object_order = object_order
        self.paths = paths
        self.transform = transform

    def train_test_split(self, split, basepaths):
        train_basepaths, val_basepaths, test_basepaths = [], [], []

        object_names = self.reverse_label_dict.keys()
        sketch_objects = np.asarray([self.label_dict[basepath] for basepath in basepaths])
        basepaths = np.asarray(basepaths)

        for object_name in object_names:
            object_basepaths = basepaths[sketch_objects == object_name].tolist()
            num_basepaths = len(object_basepaths)
            num_train = int(0.7 * num_basepaths)
            num_val = int(0.8 * num_basepaths) - num_train
            random.shuffle(object_basepaths)
            train_basepaths += object_basepaths[:num_train]
            val_basepaths += object_basepaths[num_train:num_train+num_val]
            test_basepaths += object_basepaths[num_train+num_val:]

        if split == 'train':
            paths = train_basepaths
        elif split == 'val':
            paths = val_basepaths
        else:  # split == 'test'
            paths = test_basepaths
        return paths

    def __getitem__(self, index):
        path = self.paths[index] 
        obj = self.label_dict[os.path.basename(path)]
        obj_ix = self.object_order.index(obj)
        category = OBJECT_TO_CATEGORY[obj]
        context = self.context_dict[os.path.basename(path)]
        context = 0 if context == 'closer' else 1
        sketch = torch.from_numpy(np.load(os.path.join(self.dirname, path)))

        if self.transform:
            sketch = self.transform(sketch)

        label = self.annotations[obj_ix, :, context]
        return sketch, label

    def __len__(self):
        return self.size


class ExhaustiveDataset(Dataset):
    """Used to create the RDM and JSON. Loops through every sketch & photo pair."""
    def __init__(self, layer='fc6', photo_transform=None, sketch_transform=None, random_seed=42):
        super(ExhaustiveDataset, self).__init__()
        np.random.seed(random_seed)
        random.seed(random_seed)
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % layer
        photo_dirname = os.path.join(db_path, 'photos')
        sketch_dirname = os.path.join(db_path, 'sketch')
        sketch_basepaths = os.listdir(sketch_dirname)
        # remove bad/corrupted images
        valid_game_ids = np.asarray(pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))['valid_gameids']).tolist()
        # only keep sketches that are in the valid_gameids (some games are garbage)
        sketch_basepaths = [path for path in sketch_basepaths 
                            if os.path.basename(path).split('_')[1] in valid_game_ids]
        sketch_paths = [os.path.join(sketch_dirname, path) for path in sketch_basepaths]
        # this details how labels are stored (order of objects)
        object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()
        # load all 32 of them once since for every sketch we use the same 32 photos
        photo_32_paths = [object_name + '.npy' for object_name in object_order]
        
        with open(os.path.join(db_path, 'sketchpad_context_dict.pickle')) as fp:
            self.context_dict = cPickle.load(fp)
        with open(os.path.join(db_path, 'sketchpad_label_dict.pickle')) as fp:
            self.label_dict = cPickle.load(fp)

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
        sketch = np.load(os.path.join(self.sketch_dirname, sketch_path))
        sketch = torch.from_numpy(sketch)
        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)
        return sketch, sketch_object, context, sketch_path

    def __len__(self):
        return self.size


class ExhaustiveSketchDataset(Dataset):
    def __init__(self, layer='fc6', transform=None, random_seed=42):
        super(ExhaustiveSketchDataset, self).__init__()
        np.random.seed(random_seed)
        random.seed(random_seed)
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % layer
        sketch_dirname = os.path.join(db_path, 'sketch')
        sketch_basepaths = os.listdir(sketch_dirname)
        # remove bad/corrupted images
        valid_game_ids = np.asarray(pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))['valid_gameids']).tolist()
        # only keep sketches that are in the valid_gameids (some games are garbage)
        sketch_basepaths = [path for path in sketch_basepaths
                            if os.path.basename(path).split('_')[1] in valid_game_ids]
        sketch_paths = [os.path.join(sketch_dirname, path) for path in sketch_basepaths]
        # this details how labels are stored (order of objects)
        object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()

        with open(os.path.join(db_path, 'sketchpad_context_dict.pickle')) as fp:
            self.context_dict = cPickle.load(fp)
        with open(os.path.join(db_path, 'sketchpad_label_dict.pickle')) as fp:
            self.label_dict = cPickle.load(fp)

        self.sketch_dirname = sketch_dirname
        self.size = len(sketch_paths)
        self.sketch_paths = sketch_paths
        self.object_order = object_order
        self.transform = transform

    def __getitem__(self, index):
        sketch_path = self.sketch_paths[index]
        context = self.context_dict[os.path.basename(sketch_path)]
        sketch_object = self.label_dict[os.path.basename(sketch_path)]
        sketch = np.load(os.path.join(self.sketch_dirname, sketch_path))
        sketch = torch.from_numpy(sketch)
        if self.transform:
            sketch = self.transform(sketch)
        return sketch, sketch_object, context, sketch_path

    def __len__(self):
        return self.size

