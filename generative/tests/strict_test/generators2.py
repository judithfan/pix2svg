"""This file contains more fancy generators, like ones
that drop strokes. These generators work directly on images, 
not on embeddings.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import csv
import copy
import random
from glob import glob

import numpy as np
import torch
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms

sys.path.append('../..')
from linerender import BresenhamRenderNet


class DropStrokeGenerator(object):
    """This will generate data according to the following pairs:
        
        (photo1, sketch1), 
        (photo2, sketch2), 
        (photo1, sketch2), 
        (photo2, sketch1),

    where photo2 is a photo of the same class.
    Each batch will be composed of a number of these 4-pair structs.

    Notably, across different epochs, repeated calls to this 
    generator will return different photo1, photo2 which should help
    with generalization.

    The train-test split here is made up of intra-category splits
    i.e. 80% of dog photos are used in training, and 20% of dog
    photos are used in testing.

    The sketches will be stored as a CSV of strokes. Each file in 
    this directory is a csv file of strokes. We will be randomly 
    dropping out N strokes from a random starting point with some 
    probability.
    """
    def __init__(self, photo_emb_dir, sketch_csv_dir, train=True,
                 batch_size=10, use_cuda=False, drop_proba=0.5):
        self.photo_emb_dir = photo_emb_dir
        self.sketch_csv_dir = sketch_csv_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.train = train
        self.drop_proba = drop_proba
        self.size = get_size()

        # save VGG model that will be used to get image embeddings
        vgg = models.vgg19(pretrained=True)
        vgg.eval()
        if use_cuda:
            vgg.cuda()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

        # save transformation for all images
        self.preprocessing = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

    def train_test_split(self):
        categories = glob(os.path.join(self.photo_csv_dir, '*'))
        train_photos, test_photos = [], []
        for cat in categories:
            paths = glob(os.path.join(cat, '*'))
            # 80% of the files will be used for training
            # 20% will be reserved for testing
            split = int(0.8 * len(paths))
            train_photos += paths[:split]
            test_photos += paths[split:]

        random.shuffle(train_photos)
        random.shuffle(test_photos)

        return train_photos, test_photos

    def get_size(self):
        train, test = self.train_test_split()
        return len(train) if self.train else len(test)

    def load_photo(self, path):
        photo = np.load(path)[np.newaxis, :, :]
        return torch.from_numpy(photo)

    def load_sketch(self, path, drop_proba=0.5, drop_num=1):
        stroke_info = []
        with open(path, 'rb') as fp:
            reader = csv.reader(fp)
            for row in reader:
                stroke_info.append(row)
        stroke_header = stroke_info[0]
        stroke_info = np.array(stroke_info[1:])

        if random.random() > drop_proba:
            stroke_info = self.dropstroke(stroke_header, stroke_info)

        x_list = stroke_info[:, stroke_header.index('x')].astype(np.float)
        y_list = stroke_info[:, stroke_header.index('y')].astype(np.float)
        pen_list = stroke_info[:, stroke_header.index('pen')].astype(int)

        # render the sketch using a non-differentiable renderer
        renderer = BresenhamRenderNet(x_list, y_list, pen_list=pen_list, 
                                      imsize=256, linewidth=3)
        sketch = renderer.forward()
        sketch_np = sketch.cpu().numpy()[0]
        # load it as an image in PIL so we can use the same preprocessing 
        # script that photos use
        image = Image.fromarray(sketch_np)
        image = self.preprocessing(image)

        return image


    def dropstroke(stroke_header, stroke_info, drop_num=1):
        stroke_ids = stroke_info[:, stroke_header.index('strokeID')].astype(np.int)
        n_strokes = max(stroke_ids)
        # pick start point to drop
        stroke_ix = random.choice(range(n_strokes - drop_num))

        ixkeep = np.logical_or(stroke_ids < stroke_ix,
                               stroke_ids > (stroke_ix + drop_num))
        return stroke_info[:, ixkeep]

    def gen_vgg_emb(self, x):
        extractor = list(self.vgg.features)[:22]
        for i in range(len(extractor)):
            x = extractor[i](x)
        return x

    def make_generator(self):
        dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # get a list of photo paths that we will loop through
        train_photo_paths, test_photo_paths = self.train_test_split()
        # photo_paths is going to be set of paths that we loop through
        photo_paths = train_photo_paths if self.train else test_photo_paths
        # blacklist_paths are files that we should not sample
        blacklist_paths = test_photo_paths if self.train else train_photo_paths

        batch_idx = 0 # keep track of when to start new batch

        for i in range(self.size):
            photo1_path = photo_paths[i]
            sketch1_path = sample_sketch_from_photo_path(photo1_path, self.sketch_emb_dir, 
                                                         deterministic=False)
            photo2_path = sample_photo_from_photo_path(photo1_path, self.photo_emb_dir, 
                                                       blacklist=blacklist_paths, same_class=True)
            sketch2_path = sample_sketch_from_photo_path(photo2_path, self.sketch_emb_dir, 
                                                         deterministic=False)

            # given a csv path, we will be doing the following operations:
            # - apply strokedrop
            # - render image
            # - pass through VGG to get embedding
            # - return torch object
            photo1 = self.load_photo(photo1_path)
            # sketch1 will hold the raw sketch as a processed Tensor
            sketch1 = self.load_sketch(sketch1_path, drop_proba=self.drop_proba)
            photo2 = self.load_photo(photo2_path)
            # sketch2 will hold the raw sketch as a processed Tensor
            sketch2 = self.load_sketch(sketch2_path, drop_proba=self.drop_proba)

            # stack a group together to be very explicit of certain relationships
            # between photo1 and photo2
            photo_group = tensor.stack((photo1, photo2, photo1, photo2))
            sketch_group = tensor.stack((sketch1, sketch2, sketch2, sketch1))
            label_group = torch.Tensor((1, 1, 0, 0))

            # build a batch
            if batch_idx == 0:
                photo_batch = photo_group
                sketch_batch = sketch_group
                label_batch = label_group
            else:
                photo_batch = torch.stack((photo_batch, photo_group))
                sketch_batch = torch.stack((sketch_batch, sketch_group))
                label_batch = torch.cat((label_batch, label_group))

            batch_idx += 1

            if batch_idx == self.batch_size:
                # we need to pass all the sketches through vgg
                sketch_batch = self.gen_vgg_emb(sketch_batch)
                # wrap stuff in variables and we are good to go
                photo_batch = Variable(photo_batch.type(dtype))
                sketch_batch = Variable(sketch_batch.type(dtype))
                label_batch = Variable(label_batch.type(dtype), requires_grad=False)

                yield (photo_batch, sketch_batch, label_batch)
                batch_idx = 0

        # return any remaining data
        if batch_idx > 0:
            sketch_batch = self.gen_vgg_emb(sketch_batch)
            photo_batch = Variable(photo_batch.type(dtype))
            sketch_batch = Variable(sketch_batch.type(dtype))
            label_batch = Variable(label_batch.type(dtype), requires_grad=False)

            yield (photo_batch, sketch_batch, label_batch)
