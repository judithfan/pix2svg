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


class SketchPlusGoodBadPhoto(Dataset):
    def __init__(self, layer='fc6', soft_labels=False, train=True, photo_transform=None, sketch_transform=None):
        super(SketchPlusGoodBadPhoto, self).__init__()
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % layer
        photos_path = os.path.join(db_path, 'photos')
        sketch_path = os.path.join(db_path, 'sketch')
        sketch_paths = os.listdir(sketch_path)

        valid_game_ids = np.asarray(pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))['valid_gameids']).tolist()
        # only keep sketches that are in the valid_gameids (some games are garbage)
        sketch_paths = [path for path in sketch_paths if os.path.basename(path).split('_')[1] in valid_game_ids]
        sketch_paths = [os.path.join(sketch_path, path) for path in sketch_paths]

        # this details how labels are stored (order of objects)
        object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()
        
        # load all 32 of them once since for every sketch we use the same 32 photos
        photo_paths = [os.path.join(photos_path, object_name + '.npy') for object_name in object_order]
        photos = np.vstack([np.load(os.path.join(photos_path, object_name + '.npy'))[np.newaxis, ...] 
                            for object_name in object_order])
        photos = torch.from_numpy(photos)

        # load human annotated labels.
        self.labels = np.load('/mnt/visual_communication_dataset/human_confusion.npy')
        with open(os.path.join(db_path, 'sketchpad_context_dict.pickle')) as fp:
            self.context_dict = cPickle.load(fp)

        with open(os.path.join(db_path, 'sketchpad_label_dict.pickle')) as fp:
            self.label_dict = cPickle.load(fp)

        # reverse label from string to integer
        self.reverse_label_dict = defaultdict(lambda: [])
        for path, label in self.label_dict.iteritems():
            self.reverse_label_dict[label].append(path)

        self.db_path = db_path
        self.photos = photos
        self.photo_paths = photo_paths
        self.object_order = object_order
        self.sketch_paths = sketch_paths
        self.size = len(sketch_paths)
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform
        self.sketch_dir = os.path.dirname(sketch_paths[0])
        self.soft_labels = soft_labels

    def __getitem__(self, index):
        sketch1_path = self.sketch_paths[index] 
        sketch1_object = self.label_dict[os.path.basename(sketch1_path)]
        object1_ix = self.object_order.index(sketch1_object)
        other_object = random.choice(list(set(self.object_order) - set([sketch1_object])))
        object2_ix = self.object_order.index(other_object)
        photo1_path = self.photo_paths[object1_ix]
        photo2_path = self.photo_paths[object2_ix]

        sketch1 = np.load(sketch1_path)    # current sketch
        sketch1 = torch.from_numpy(sketch1)
        photo1 = self.photos[object1_ix]   # photo of current sketch
        photo2 = self.photos[object2_ix]   # random photo that's not the current photo

        if self.sketch_transform:
            sketch1 = self.sketch_transform(sketch1)

        if self.photo_transform:
            photo1 = self.photo_transform(photo1)
            photo2 = self.photo_transform(photo2)

        if self.soft_labels:
            context1 = self.context_dict[os.path.basename(sketch1_path)]
            context1 = 0 if context1 == 'closer' else 1
            soft_label1 = torch.from_numpy(self.labels[object1_ix, :, context1]).float()
            return sketch1, photo1, photo2, object1_ix, object2_ix, soft_label1
            
        return sketch1, photo1, photo2, object1_ix, object2_ix, None

    def __len__(self):
        return self.size


class SketchPlus32Photos(Dataset):
    """Assumes that we have precomputed conv-4-2 layer embeddings."""
    def __init__(self, layer='fc6', train=True, return_paths=False, photo_transform=None, sketch_transform=None):
        super(SketchPlus32Photos, self).__init__()
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % layer
        photos_path = os.path.join(db_path, 'photos')
        sketch_path = os.path.join(db_path, 'sketch')
        sketch_paths = os.listdir(sketch_path)

        valid_game_ids = np.asarray(pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))['valid_gameids']).tolist()
        # only keep sketches that are in the valid_gameids (some games are garbage)
        sketch_paths = [path for path in sketch_paths if os.path.basename(path).split('_')[1] in valid_game_ids]
        sketch_paths = [os.path.join(sketch_path, path) for path in sketch_paths]                                                                                                                                                                   # this details how labels are stored (order of objects)
        object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()

        # load all 32 of them once since for every sketch we use the same 32 photos
        photo_paths = [os.path.join(photos_path, object_name + '.npy') for object_name in object_order]
        photos = np.vstack([np.load(os.path.join(photos_path, object_name + '.npy'))[np.newaxis, ...]
                            for object_name in object_order])
        photos = torch.from_numpy(photos)

        # load human annotated labels.
        self.labels = np.load('/mnt/visual_communication_dataset/human_confusion.npy')
        with open(os.path.join(db_path, 'sketchpad_context_dict.pickle')) as fp:
            self.context_dict = cPickle.load(fp)

        with open(os.path.join(db_path, 'sketchpad_label_dict.pickle')) as fp:
            self.label_dict = cPickle.load(fp)

        self.db_path = db_path
        self.photos = photos
        self.photo_paths = photo_paths
        self.object_order = object_order
        self.sketch_paths = sketch_paths
        self.size = len(sketch_paths)
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform
        self.return_paths = return_paths

    def __getitem__(self, index):
        photos = self.photos
        photo_paths = self.photo_paths
        if self.photo_transform is not None:
            photos = self.photo_transform(photos)
        # load a single sketch path
        sketch_path = self.sketch_paths[index]
        # use sketch path to find context
        context = self.context_dict[os.path.basename(sketch_path)]
        context = 0 if context == 'closer' else 1
        # use sketch path to pull the correct label vector
        sketch_object = self.label_dict[os.path.basename(sketch_path)]
        sketch_ix = self.object_order.index(sketch_object)
        label = self.labels[sketch_ix, :, context]
        label = torch.from_numpy(label).float()
        # load the physical sketch conv-4-2 embedding
        sketch = np.load(sketch_path)
        sketch = torch.from_numpy(sketch)
        if self.sketch_transform is not None:
            sketch = self.sketch_transform(sketch)

        if self.return_paths:
            return photos, sketch, photo_paths, sketch_path, label
        return photos, sketch, label

    def __len__(self):
        return self.size


