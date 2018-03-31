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
 

class SketchPlus32PhotosHARD(SketchPlus32Photos):
    def __init__(self, layer='fc6', train=True, return_paths=False, photo_transform=None, sketch_transform=None):
        super(SketchPlus32PhotosHARD, self).__init__(layer=layer, train=train, return_paths=return_paths,
                                                     photo_transform=photo_transform, 
                                                     sketch_transform=sketch_transform)
        self.hard_labels = np.load(os.path.join(self.db_path, 'sketchpad_hard_labels.npy'))
        self.sketch_dir = os.path.dirname(self.sketch_paths[0])
    
    def __getitem__(self, index):
        photos = self.photos
        photo_paths = self.photo_paths
        if self.photo_transform is not None:
            photos = self.photo_transform(photos)
    
        sketch_path, sketch_label = self.hard_labels[index]
        sketch_path = os.path.join(self.sketch_dir, sketch_path)
        label = int(sketch_label)

        # load the physical sketch conv-4-2 embedding
        sketch = np.load(sketch_path)
        sketch = torch.from_numpy(sketch)
        if self.sketch_transform is not None:
            sketch = self.sketch_transform(sketch)

        if self.return_paths:
            return photos, sketch, photo_paths, sketch_path, label
        return photos, sketch, label

    def __len__(self):
        return len(self.hard_labels)


class SketchPlus32PhotosCATEGORY(SketchPlus32Photos):
    def __init__(self, layer='fc6', train=True, return_paths=False, photo_transform=None, sketch_transform=None):
        super(SketchPlus32PhotosCATEGORY, self).__init__(layer=layer, train=train, return_paths=return_paths,
                                                         photo_transform=photo_transform,
                                                         sketch_transform=sketch_transform)
        self.reverse_label_dict = defaultdict(lambda: [])
        for path, label in self.label_dict.iteritems():
            self.reverse_label_dict[label].append(path)
        self.sketch_dir = os.path.dirname(self.sketch_paths[0])

    def __getitem__(self, index):
        sketch1_path = self.sketch_paths[index] 
        sketch1_object = self.label_dict[os.path.basename(sketch1_path)]
        sketch1_object_ix = self.object_order.index(sketch1_object)
        sketch2_object = random.choice(list(set(self.object_order) - set([sketch1_object])))
        sketch2_object_ix = self.object_order.index(sketch2_object)
        sketch2_path = random.choice(self.reverse_label_dict[sketch2_object])
        photo1_path = self.photo_paths[sketch1_object_ix]
        photo2_path = self.photo_paths[sketch2_object_ix]

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
        photo1 = self.photos[sketch1_object_ix]   # photo of current sketch
        photo2 = self.photos[sketch2_object_ix]   # random photo that's not the current photo

        # find category for sketch1 and sketch2
        category1 = self.labels[sketch1_object_ix, :, context1]
        category2 = self.labels[sketch2_object_ix, :, context2]
        category1 = torch.from_numpy(category1).float()
        category2 = torch.from_numpy(category2).float()        
        # category1 = sketch1_object_ix
        # category2 = sketch2_object_ix

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
        label_group = torch.Tensor([1, 1, 0, 0])
        category_group = torch.cat([category1.unsqueeze(0), category2.unsqueeze(0), 
                                    category2.unsqueeze(0), category1.unsqueeze(0)], dim=0)

        if self.return_paths:
            return photo_group, sketch_group, label_group, category_group, \
                   [photo1_path, photo2_path], [sketch1_path, sketch2_path]
        return photo_group, sketch_group, label_group, category_group
 

class SketchPlus32PhotosSOFT(SketchPlus32Photos):
    def __init__(self, layer='fc6', train=True, return_paths=False, photo_transform=None, sketch_transform=None):
        super(SketchPlus32PhotosSOFT, self).__init__(layer=layer, train=train, return_paths=return_paths,
                                                     photo_transform=photo_transform,
                                                     sketch_transform=sketch_transform)
        self.reverse_label_dict = defaultdict(lambda: [])
        for path, label in self.label_dict.iteritems():
            self.reverse_label_dict[label].append(path)
        self.sketch_dir = os.path.dirname(self.sketch_paths[0])

    def __getitem__(self, index):
        sketch1_path = self.sketch_paths[index] 
        sketch1_object = self.label_dict[os.path.basename(sketch1_path)]
        sketch1_object_ix = self.object_order.index(sketch1_object)
        sketch2_object = random.choice(list(set(self.object_order) - set([sketch1_object])))
        sketch2_object_ix = self.object_order.index(sketch2_object)
        sketch2_path = random.choice(self.reverse_label_dict[sketch2_object])
        photo1_path = self.photo_paths[sketch1_object_ix]
        photo2_path = self.photo_paths[sketch2_object_ix]

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
        photo1 = self.photos[sketch1_object_ix]   # photo of current sketch
        photo2 = self.photos[sketch2_object_ix]   # random photo that's not the current photo

        # find category for sketch1 and sketch2
        # category1 = self.labels[sketch1_object_ix, :, context1]
        # category2 = self.labels[sketch2_object_ix, :, context2]
        # category1 = torch.from_numpy(category1).float()
        # category2 = torch.from_numpy(category2).float() 
        category1 = sketch1_object_ix
        category2 = sketch2_object_ix

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
        label_group = torch.Tensor([self.labels[sketch1_object_ix, sketch1_object_ix, context1],
                                    self.labels[sketch2_object_ix, sketch2_object_ix, context2],
                                    self.labels[sketch2_object_ix, sketch1_object_ix, context2],
                                    self.labels[sketch1_object_ix, sketch2_object_ix, context1]])
        category_group = torch.Tensor([category1, category2, category2, category1]).long()
        # category_group = torch.cat([category1.unsqueeze(0), category2.unsqueeze(0),
        #                             category2.unsqueeze(0), category1.unsqueeze(0)], dim=0)

        if self.return_paths:
            return photo_group, sketch_group, label_group, category_group, \
                   [photo1_path, photo2_path], [sketch1_path, sketch2_path]
        return photo_group, sketch_group, label_group, category_group


class SketchPlus32PhotosRAW(Dataset):
    """Assumes that we have precomputed conv-4-2 layer embeddings."""
    def __init__(self, train=True, return_paths=False, photo_transform=None, sketch_transform=None):
        super(SketchPlus32PhotosRAW, self).__init__()
        db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96'
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

        # load human annotated labels.
        self.labels = np.load('/mnt/visual_communication_dataset/human_confusion.npy')
        with open(os.path.join(db_path, 'sketchpad_context_dict.pickle')) as fp:
            self.context_dict = cPickle.load(fp)

        with open(os.path.join(db_path, 'sketchpad_label_dict.pickle')) as fp:
            self.label_dict = cPickle.load(fp)

        self.db_path = db_path
        self.photo_paths = photo_paths
        self.object_order = object_order
        self.sketch_paths = sketch_paths
        self.size = len(sketch_paths)
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform
        self.return_paths = return_paths

        self.reverse_label_dict = defaultdict(lambda: [])
        for path, label in self.label_dict.iteritems():
            self.reverse_label_dict[label].append(path)
        self.sketch_dir = os.path.dirname(self.sketch_paths[0])

    def __getitem__(self, index):
        sketch1_path = self.sketch_paths[index] 
        sketch1_object = self.label_dict[os.path.basename(sketch1_path).replace('.png', '.npy')]
        sketch1_object_ix = self.object_order.index(sketch1_object)
        sketch2_object = random.choice(list(set(self.object_order) - set([sketch1_object])))
        sketch2_object_ix = self.object_order.index(sketch2_object)
        sketch2_path = random.choice(self.reverse_label_dict[sketch2_object]).replace('.npy', '.png')
        photo1_path = self.photo_paths[sketch1_object_ix].replace('.npy', '.png')
        photo2_path = self.photo_paths[sketch2_object_ix].replace('.npy', '.png')

        # find context of sketch by path
        context1 = self.context_dict[os.path.basename(sketch1_path).replace('.png', '.npy')]
        context2 = self.context_dict[os.path.basename(sketch2_path).replace('.png', '.npy')]
        context1 = 0 if context1 == 'closer' else 1
        context2 = 0 if context2 == 'closer' else 1

        def process_sketch(sketch):
            sketch = alpha_composite_with_color(sketch)
            return sketch.convert('L')

        def process_photo(photo):
            return photo.convert('RGB')

        sketch2_path = os.path.join(self.sketch_dir, sketch2_path)
        sketch1 = Image.open(sketch1_path)        # current sketch
        sketch2 = Image.open(sketch2_path)        # random sketch matching with photo 2
        photo1 = Image.open(photo1_path)          # photo of current sketch
        photo2 = Image.open(photo2_path)          # random photo that's not the current photo

        sketch1 = process_sketch(sketch1)
        sketch2 = process_sketch(sketch2)
        photo1 = process_photo(photo1)
        photo2 = process_photo(photo2)

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
        label_group = torch.Tensor([self.labels[sketch1_object_ix, sketch1_object_ix, context1],
                                    self.labels[sketch2_object_ix, sketch2_object_ix, context2],
                                    self.labels[sketch2_object_ix, sketch1_object_ix, context2],
                                    self.labels[sketch1_object_ix, sketch2_object_ix, context1]])
        
        if self.return_paths:
            return photo_group, sketch_group, label_group, \
                   [photo1_path, photo2_path], [sketch1_path, sketch2_path]
        return photo_group, sketch_group, label_group

    def __len__(self):
        return self.size


class SketchPlus32PhotosRAWRDM(Dataset):
    """Assumes that we have precomputed conv-4-2 layer embeddings."""
    def __init__(self, train=True, return_paths=False, photo_transform=None, sketch_transform=None):
        super(SketchPlus32PhotosRAWRDM, self).__init__()
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
        photo_paths = [os.path.join(photos_path, object_name + '.png') for object_name in object_order]

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

        def process_sketch(sketch):
            sketch = alpha_composite_with_color(sketch)
            return sketch.convert('L')

        def process_photo(photo):
            return photo.convert('RGB')

        photo_paths = self.photo_paths
        photos = []
        for photo_path in photo_paths:
            photo = Image.open(photo_path)
            photo = process_photo(photo)
            if self.photo_transform is not None:
                photo = self.photo_transform(photo)
            photos.append(photo)
        photos = torch.stack(photos)
        # load a single sketch path
        sketch_path = self.sketch_paths[index]
        # use sketch path to find context
        context = self.context_dict[os.path.basename(sketch_path).replace('.png', '.npy')]
        context = 0 if context == 'closer' else 1
        # use sketch path to pull the correct label vector
        sketch_object = self.label_dict[os.path.basename(sketch_path).replace('.png', '.npy')]
        sketch_ix = self.object_order.index(sketch_object)
        label = self.labels[sketch_ix, :, context]
        label = torch.from_numpy(label).float()
        # load the physical sketch conv-4-2 embedding
        sketch = Image.open(sketch_path)
        sketch = process_sketch(sketch)
        if self.sketch_transform is not None:
            sketch = self.sketch_transform(sketch)

        if self.return_paths:
            return photos, sketch, photo_paths, sketch_path, label
        return photos, sketch, label

    def __len__(self):
        return self.size


def alpha_composite(front, back):
    """Alpha composite two RGBA images.
    Source: http://stackoverflow.com/a/9166671/284318
    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object
    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result


def alpha_composite_with_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.
    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)
    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)

