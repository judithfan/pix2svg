from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import random
from glob import glob

import torch
import numpy as np
from PIL import Image

from utils import list_files


preprocessing = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])


class ReferenceGameGenerator(object):
    """This generates pairs from the reference game data. 
    This is not used for training purposes. This will yield a pair 
    (sketch, render) for every sketch and every render, so it will 
    be a total of |sketch| * |render| images.

    :param image_dir: path to folder of images
    :param use_cuda: whether to return cuda.FloatTensor or FloatTensor
    """

    def __init__(self, sketch_dir, render_dir, use_cuda=False):
        self.sketch_dir = sketch_dir
        self.render_dir = render_dir
        self.dtype = dtype = (torch.cuda.FloatTensor 
                              if use_cuda else torch.FloatTensor)

    def make_generator(self):
        sketch_paths = [path for path in list_files(self.sketch_dir, ext='png')]
        render_paths = [path for path in list_files(self.render_dir, ext='png')]

        n_sketches = len(sketch_paths)
        n_renders = len(render_paths)
        self.size = n_sketches * n_renders

        for i in range(n_sketches):
            for j in range(n_renders):
                sketch_path, render_path = sketch_paths[i], render_paths[j]
                sketch = Image.open(sketch_path)
                render = Image.open(render_path)

                sketch = preprocessing(sketch).unsqueeze(0)
                render = preprocessing(render).unsqueeze(0)

                sketch = Variable(sketch.type(self.dtype), volatile=True)
                render = Variable(render.type(self.dtype), volatile=True)

                yield (sketch_path, sketch, render_path, render)


def list_files(path, ext='jpg'):
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result

