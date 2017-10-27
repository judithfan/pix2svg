from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import random
import itertools
from glob import glob

import torch
import numpy as np
from PIL import Image

from torch.autograd import Variable
import torchvision.transforms as transforms

from generators import preprocessing, alpha_composite_with_color


class ReferenceGame2EmbeddingGenerator(object):
    """This generates pairs from the reference game data. 
    This is not used for training purposes. This will yield a pair 
    (sketch, render) for every sketch and every render, so it will 
    be a total of |sketch| * |render| images.

    :param sketch_dir: path to folder of sketch pngs
    :param render_dir: path to folder of image pngs
    :param use_cuda: whether to return cuda.FloatTensor or FloatTensor
    """

    def __init__(self, use_cuda=False):
        self.data_dir = '/data/jefan/sketchpad_basic_fixedpose_conv_4_2'
        self.dtype = dtype = (torch.cuda.FloatTensor 
                              if use_cuda else torch.FloatTensor)

        with open(os.path.join(self.data_dir, './incorrect_trial_paths.txt')) as fp:
            bad_games = fp.readlines()
            bad_games = [os.path.join(self.data_dir, 'sketch', 
                                      i.replace('.png\n', '.npy')) 
                         for i in bad_games]

        sketch_paths = glob(os.path.join(self.data_dir, 'sketch', '*.npy'))
        sketch_paths = list(set(sketch_paths) - set(bad_games))
        render_paths = glob(os.path.join(self.data_dir, 'target', '*.npy')) + \
                       glob(os.path.join(self.data_dir, 'distractor1', '*.npy')) + \
                       glob(os.path.join(self.data_dir, 'distractor2', '*.npy')) + \
                       glob(os.path.join(self.data_dir, 'distractor3', '*.npy'))
        self.generator = itertools.product(sketch_paths, render_paths)
        self.size = len(sketch_paths) * len(render_paths)

    def make_generator(self):
        generator = self.generator

        while True:
            try:
                sketch_path, render_path = next(generator)
                sketch = np.load(sketch_path)
                render = np.load(render_path)
                sketch = torch.from_numpy(sketch).unsqueeze(0)
                render = torch.from_numpy(render).unsqueeze(0)
                sketch = Variable(sketch.type(self.dtype), volatile=True)
                render = Variable(render.type(self.dtype), volatile=True)
                yield (sketch_path, sketch, render_path, render)
            except StopIteration:
                break
