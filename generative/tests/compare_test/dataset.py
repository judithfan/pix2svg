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
 

class VisualCommunicationDataset(Dataset):
    """Used for training. We show a group of 4 images at a time."""
    def __init__(self, layer='fc6', split='train', soft_labels=False, photo_transform=None,
                 sketch_transform=None, random_seed=42, alpha=.5):
        super(Dataset, self).__init__()
        np.random.seed(random_seed)
        random.seed(random_seed)
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % layer
        photo_dirname = os.path.join(db_path, 'photos')
        sketch_dirname = os.path.join(db_path, 'sketch')
        sketch_basepaths = os.listdir(sketch_dirname)

        valid_game_ids = np.asarray(pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))['valid_gameids']).tolist()
        # only keep sketches that are in the valid_gameids (some games are garbage)
        sketch_basepaths = [path for path in sketch_basepaths if os.path.basename(path).split('_')[1] in valid_game_ids]

        # this details how labels are stored (order of objects)
        object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()

        # load all 32 of them once since for every sketch we use the same 32 photos
        photo_32_paths = [object_name + '.npy' for object_name in object_order]

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
        self.sketch_dirname = sketch_dirname 
        self.photo_dirname = photo_dirname
        self.size = len(sketch_paths)
        self.split = split
        self.alpha = alpha
        self.use_soft_labels = soft_labels
        self.photo_32_paths = photo_32_paths
        self.object_order = object_order
        self.sketch_paths = sketch_paths
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform

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
            num_train = int(0.7 * num_basepaths)
            num_val = int(0.8 * num_basepaths) - num_train
            random.shuffle(object_basepaths)
            train_sketch_basepaths += object_basepaths[:num_train]
            val_sketch_basepaths += object_basepaths[num_train:num_train+num_val]
            test_sketch_basepaths += object_basepaths[num_train+num_val:]

        if split == 'train':
            sketch_paths = train_sketch_basepaths
        elif split == 'val':
            sketch_paths = val_sketch_basepaths
        else:  # split == 'test'
            sketch_paths = test_sketch_basepaths
        return sketch_paths

    def __getitem__(self, index):
        sketch1_path = self.sketch_paths[index] 
        sketch1_object = self.label_dict[os.path.basename(sketch1_path)]
        sketch1_object_ix = self.object_order.index(sketch1_object)
        alpha_sample = np.random.binomial(1, self.alpha)
        sketch1_category = OBJECT_TO_CATEGORY[sketch1_object]
        sketch2_objects = CATEGORY_TO_OBJECT[sketch1_category]
        if alpha_sample == 1:
            sketch2_object = random.choice(list(set(sketch2_objects) - set([sketch1_object])))
        else:
            sketch2_object = random.choice(list(set(self.object_order) - set(sketch2_objects)))
        sketch2_object_ix = self.object_order.index(sketch2_object)
        sketch2_path = random.choice(list(set(self.reverse_label_dict[sketch2_object]).intersection(set(self.sketch_paths))))
        photo1_path = self.photo_32_paths[sketch1_object_ix]
        photo2_path = self.photo_32_paths[sketch2_object_ix]

        # find context of sketch by path
        context1 = self.context_dict[os.path.basename(sketch1_path)]
        context2 = self.context_dict[os.path.basename(sketch2_path)]
        context1 = 0 if context1 == 'closer' else 1
        context2 = 0 if context2 == 'closer' else 1

        sketch1 = torch.from_numpy(np.load(os.path.join(self.sketch_dirname, sketch1_path)))
        sketch2 = torch.from_numpy(np.load(os.path.join(self.sketch_dirname, sketch2_path)))
        photo1 = torch.from_numpy(np.load(os.path.join(self.photo_dirname, photo1_path)))
        photo2 = torch.from_numpy(np.load(os.path.join(self.photo_dirname, photo2_path)))

        if self.sketch_transform:
            sketch1 = self.sketch_transform(sketch1)
            sketch2 = self.sketch_transform(sketch2)

        if self.photo_transform:
            photo1 = self.photo_transform(photo1)
            photo2 = self.photo_transform(photo2)

        photo_group = torch.cat((photo1.unsqueeze(0), photo2.unsqueeze(0), 
                                 photo1.unsqueeze(0), photo2.unsqueeze(0)), dim=0)
        sketch_group = torch.cat((sketch1.unsqueeze(0), sketch2.unsqueeze(0), 
                                  sketch2.unsqueeze(0), sketch1.unsqueeze(0)), dim=0)
        if self.use_soft_labels:
            label_group = torch.FloatTensor([self.annotations[sketch1_object_ix, sketch1_object_ix, context1],
                                             self.annotations[sketch2_object_ix, sketch2_object_ix, context2],
                                             self.annotations[sketch2_object_ix, sketch1_object_ix, context2],
                                             self.annotations[sketch1_object_ix, sketch2_object_ix, context1]])
        else:
            label_group = torch.Tensor([1, 1, 0, 0])
        photo_class = torch.LongTensor([sketch1_object_ix, sketch2_object_ix, sketch1_object_ix, sketch2_object_ix])
        sketch_class = torch.LongTensor([sketch1_object_ix, sketch2_object_ix, sketch2_object_ix, sketch1_object_ix])
        return photo_group, sketch_group, label_group.unsqueeze(1), photo_class, sketch_class

    def __len__(self):
        return self.size


class HumanAnnotationDataset(VisualCommunicationDataset):
    def __getitem__(self, index):
        sketch_path = self.sketch_paths[index] 
        sketch_object = self.label_dict[os.path.basename(sketch_path)]
        sketch_object_ix = self.object_order.index(sketch_object)
        sketch_context = self.context_dict[os.path.basename(sketch_path)]
        sketch_context = 0 if sketch_context == 'closer' else 1

        sketch = torch.from_numpy(np.load(os.path.join(self.sketch_dirname, sketch_path)))
        photo_32 = torch.stack([torch.from_numpy(np.load(os.path.join(self.photo_dirname, photo_path)))
                                for photo_path in self.photo_32_paths])

        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)

        if self.photo_transform:
            photo_32 = self.photo_transform(photo_32)

        if self.use_soft_labels:
            label = torch.from_numpy(self.annotations[sketch_object_ix, :, context1])
        else:
            label = torch.zeros(32)
            label[sketch_object_ix] = 1
        return photo_32, sketch, label


class SketchOnlyDataset(Dataset):
    def __init__(self, layer='fc6', split='train', soft_labels=False, transform=None, random_seed=42):
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
        self.use_soft_labels = soft_labels
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
        label = self.object_order.index(obj)
        context = self.context_dict[os.path.basename(path)]
        context = 0 if context == 'closer' else 1
        sketch = torch.from_numpy(np.load(os.path.join(self.dirname, path)))

        if self.transform:
            sketch = self.transform(sketch)

        if self.use_soft_labels:
            label = self.annotations[obj_ix, :, context]

        return sketch, label

    def __len__(self):
        return self.size


class ExhaustiveDataset(Dataset):
    """Used to create the RDM and JSON. Loops through every sketch & photo pair."""
    def __init__(self, layer='fc6', photo_transform=None, sketch_transform=None, random_seed=42):
        super(Dataset, self).__init__()
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
