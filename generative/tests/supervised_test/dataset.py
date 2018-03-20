from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cPickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset


class SketchPlus32Photos(Dataset):
    """Assumes that we have precomputed conv-4-2 layer embeddings."""
    def __init__(self, train=True, photo_transform=None, sketch_transform=None):
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_conv_4_2/'
        photos_path = os.path.join(db_path, 'photos')
        sketch_path = os.path.join(db_path, 'sketch')
        sketch_paths = os.listdir(sketch_path)

        valid_game_ids = np.asarray(pd.read_csv('/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_conv_4_2/valid_gameids_pilot2.csv')['valid_gameids']).tolist()
        # only keep sketches that are in the valid_gameids (some games are garbage)
        sketch_paths = [path for path in sketch_paths if os.path.basename(path).split('_')[1] in valid_game_ids]

        # this details how labels are stored (order of objects)
        object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
        object_order = np.asarray(object_order['object_name']).tolist()
        
        # load all 32 of them once since for every sketch we use the same 32 photos
        photos = np.vstack([np.load(os.path.join(photos_path, object_name)) 
                            for object_name in object_order])
        photos = torch.from_numpy(photos)

        # load human annotated labels.
        self.labels = np.load('/mnt/visual_communication_dataset/human_confusion.npy')
        with open('/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_conv_4_2/sketchpad_context_dict.pickle') as fp:
            self.context_dict = cPickle.load(fp)       

        self.photos = photos
        self.object_order = object_order
        self.sketch_paths = sketch_paths
        self.size = len(sketch_paths)
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform

    def __getitem__(self, index):
        photos = self.photos
        if self.photo_transform is not None:
            photos = self.photo_transform(photos)
        
        # load a single sketch path
        sketch_path = self.sketch_paths[index]
        # use sketch path to find context
        context = context_dict[os.path.basename(sketch_path)
        context = 0 if context == 'closer' else 'further'
        # use sketch path to pull the correct label vector
        sketch_object = os.path.splitext(os.path.basename(sketch_path).split('_')[-1])[0]
        sketch_ix = self.object_order.index(sketch_object)
        label = self.label[sketch_ix, :, context]
        # load the physical sketch conv-4-2 embedding
        sketch = np.load(sketch_path)
        sketch = torch.from_numpy(sketch)
        if self.sketch_transform is not None:
            sketch = self.sketch_transform(sketch)

        return photos, sketch, label
    
    def __len__(self):
        return self.size
 
