from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cPickle
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data.dataset import Dataset


class SketchPlus32Photos(Dataset):
    def __init__(self, layer='fc6', train=True, return_paths=False, soft_labels=False,
                 photo_transform=None, sketch_transform=None):
        super(SketchPlus32Photos, self).__init__()
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
        self.return_paths = return_paths
        self.soft_labels = soft_labels
        self.sketch_dir = os.path.dirname(sketch_paths[0])

    def __getitem__(self, index):
        sketch1_path = self.sketch_paths[index] 
        sketch1_object = self.label_dict[os.path.basename(sketch1_path)]
        object1_ix = self.object_order.index(sketch1_object)
        sketch2_object = random.choice(list(set(self.object_order) - set([sketch1_object])))
        object2_ix = self.object_order.index(sketch2_object)
        sketch2_path = random.choice(self.reverse_label_dict[sketch2_object])
        photo1_path = self.photo_paths[object1_ix]
        photo2_path = self.photo_paths[object2_ix]

        # find context of sketch by path
        context1 = self.context_dict[os.path.basename(sketch1_path)]
        context2 = self.context_dict[os.path.basename(sketch2_path)]
        context1 = 0 if context1 == 'closer' else 1
        context2 = 0 if context2 == 'closer' else 1

        sketch2_path = os.path.join(self.sketch_dir, sketch2_path)

        sketch1 = np.load(sketch1_path)           # current sketch
        sketch2 = np.load(sketch2_path)           # random sketch matching with photo 2
        sketch1 = torch.from_numpy(sketch1)
        sketch2 = torch.from_numpy(sketch2)
        photo1 = self.photos[object1_ix]   # photo of current sketch
        photo2 = self.photos[object2_ix]   # random photo that's not the current photo

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
        if self.soft_labels:
            label_group = torch.Tensor([self.labels[object1_ix, object1_ix, context1],
                                        self.labels[object2_ix, object2_ix, context2],
                                        self.labels[object2_ix, object1_ix, context2],
                                        self.labels[object1_ix, object2_ix, context1]])
        else:
            label_group = torch.Tensor([1, 1, 0, 0]).float()

        if self.return_paths:
            return (photo_group, sketch_group, label_group, 
                    [photo1_path, photo2_path], [sketch1_path, sketch2_path])
        return photo_group, sketch_group, label_group

    def __len__(self):
        return self.size
 

class SketchPlus32PhotosStream(SketchPlus32Photos):
    def __init__(self, layer='fc6', train=True, return_paths=False, soft_labels=False,
                 photo_transform=None, sketch_transform=None):
        super(SketchPlus32PhotosStream, self).__init__(
            layer=layer, train=train, return_paths=return_paths, soft_labels=soft_labels, 
            photo_transform=photo_transform, sketch_transform=sketch_transform)
        self._sketch_paths = copy.deepcopy(self.sketch_paths)
        self._photo_paths = copy.deepcopy(self.photo_paths)
        self.annotations = copy.deepcopy(self.labels)
        self.negative_sample()

    def negative_sample(self):
        pos_sketch_paths = copy.deepcopy(self._sketch_paths)
        neg_sketch_paths = []
        pos_photo_paths = []
        neg_photo_paths = []
        
        print('Building negative samples.')
        for i in xrange(tqdm(len(pos_sketch_paths))):
            sketch_path = pos_sketch_paths[i]
            object1 = self.label_dict[os.path.basename(sketch_path)]
            object1_ix = self.object_order.index(object1)
            object2 = random.choice(list(set(self.object_order) - set([object1])))
            object2_ix = self.object_order.index(object2)
            neg_sketch_path = random.choice(self.reverse_label_dict[object2])
            neg_sketch_paths.append(neg_sketch_path)
            pos_photo_paths.append(self._photo_paths[object1_ix])
            neg_photo_paths.append(self._photo_paths[object2_ix])

        sketch_paths = pos_sketch_paths + neg_sketch_paths
        photo_paths = pos_photo_paths + neg_photo_paths
        hard_labels = ([1 for _ in xrange(len(pos_sketch_paths))] + 
                       [0 for _ in xrange(len(neg_sketch_paths))])
        self.sketch_paths = sketch_paths
        self.photo_paths = photo_paths
        self.hard_labels = hard_labels
        self.size = len(sketch_paths)

    def __getitem__(self, index):
        sketch_path = self.sketch_paths[index]
        photo_path = self.photo_paths[index]

        sketch_object_name = self.label_dict[os.path.basename(sketch_path)]
        sketch_object_ix = self.object_order.index(sketch_object_name)
        photo_object_name = self.label_dict[os.path.basename(photo_path)]
        photo_object_ix = self.object_order.index(photo_object_name)

        sketch = np.load(sketch_path)
        sketch = torch.from_numpy(sketch)
        photo = np.load(photo_path)
        photo = torch.from_numpy(photo)

        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)

        if self.photo_transform:
            photo = self.photo_transform(photo)

        if self.soft_labels:
            context = self.context_dict[os.path.basename(sketch_path)]
            label = self.annotations[sketch_object_ix, photo_object_ix, context]
        else:
            label = self.hard_labels[index]

        if self.return_paths:
            return photo, sketch, label, photo_path, sketch_path
        return photo, sketch, label

    def __len__(self):
        return self.size
