from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from glob import glob

import numpy as np
from PIL import Image

from distribtest import list_files


def get_bbox_area(path):
    im = Image.open(path)
    im = im.convert('RGB')
    im = np.asarray(im)
    im = im[: , :, 0]

    x, y = np.where(im == 0)
    min_x = min(x)
    max_x = max(y)
    min_y = min(y)
    max_y = max(y)

    area = (max_x - min_x) * (max_y - min_y)
    return area



if __name__ == '__main__':
    losses = np.load('./outputs/loss_conv_4_2_euclidean.npy')[:-1]
    sketch_paths = list_files('/home/jefan/full_sketchy_dataset/sketches', ext='png')
    n_paths = len(sketch_paths)
    
    sketch_areas = np.zeros(n_paths)
    for i, sketch_path in enumerate(sketch_paths):
        print('Processing [{}/{}]'.format(i + 1, n_paths))
        area = get_bbox_area(sketch_path)
        sketch_areas[i] = area

    np.save('./outputs/sketch_area_conv_4_2_euclidean.npy')
    