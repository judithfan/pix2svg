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

from torch.autograd import Variable
import torchvision.transforms as transforms

preprocessing = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])


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


class ReferenceGameGenerator(object):
    """This generates pairs from the reference game data. 
    This is not used for training purposes. This will yield a pair 
    (sketch, render) for every sketch and every render, so it will 
    be a total of |sketch| * |render| images.

    :param sketch_dir: path to folder of sketch pngs
    :param render_dir: path to folder of image pngs
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

                # sketches in this dataset are transparent so we need to 
                # fill it up with a blank color: white. 
                # renderings do not have a similar problem.
                sketch = alpha_composite_with_color(sketch)

                # now convert to RGB
                sketch = sketch.convert('RGB')
                render = render.convert('RGB')

                sketch = preprocessing(sketch).unsqueeze(0)
                render = preprocessing(render).unsqueeze(0)

                sketch = Variable(sketch.type(self.dtype), volatile=True)
                render = Variable(render.type(self.dtype), volatile=True)

                yield (sketch_path, sketch, render_path, render)


class ReferenceGameEmbeddingGenerator(object):
    """This generates pairs from the reference game data. 
    This is not used for training purposes. This will yield a pair 
    (sketch, render) for every sketch and every render, so it will 
    be a total of |sketch| * |render| images.

    :param sketch_dir: path to folder of sketch pngs
    :param render_dir: path to folder of image pngs
    :param use_cuda: whether to return cuda.FloatTensor or FloatTensor
    """

    def __init__(self, sketch_dir, render_dir, use_cuda=False):
        self.sketch_dir = sketch_dir
        self.render_dir = render_dir
        self.dtype = dtype = (torch.cuda.FloatTensor 
                              if use_cuda else torch.FloatTensor)

    def make_generator(self):
        sketch_paths = [path for path in list_files(self.sketch_dir, ext='npy')]
        render_paths = [path for path in list_files(self.render_dir, ext='npy')]

        n_sketches = len(sketch_paths)
        n_renders = len(render_paths)
        self.size = n_sketches * n_renders

        for i in range(n_sketches):
            for j in range(n_renders):
                # here we are just looping through each pair of sketch and rendering
                sketch_path, render_path = sketch_paths[i], render_paths[j]
                sketch = np.load(sketch_path)
                render = np.load(render_path)

                sketch = torch.from_numpy(sketch).unsqueeze(0)
                render = torch.from_numpy(render).unsqueeze(0)

                sketch = Variable(sketch.type(self.dtype), volatile=True)
                render = Variable(render.type(self.dtype), volatile=True)

                yield (sketch_path, sketch, render_path, render)


def list_files(path, ext='jpg'):
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result

