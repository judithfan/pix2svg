from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
sys.path.append('../..')
from PIL import Image

import torch

from linerender import BresenhamRenderNet


def random_sketch(imsize=224, n_strokes=10):
    points = np.random.choice(range(imsize), n_strokes*2).reshape(n_strokes, 2)
    x_list, y_list = points[:, 0], points[:, 1]
    sketch = BresenhamRenderNet(x_list, y_list, imsize=imsize, linewidth=5).forward()
    sketch = torch.cat((sketch, sketch, sketch), dim=1)
    return (1 - sketch) * 255


def save_sketch(sketch, path):
    x = sketch.numpy().astype(np.uint8)[0]
    im = Image.fromarray(np.rollaxis(x, 0, 3))
    im.save(path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_strokes', type=int, default=10)
    parser.add_argument('--imsize', type=int, default=256)
    args = parser.parse_args()

    # given a directory of images, make another directory of 
    # noise sketches.

    photo_dir = '/home/jefan/full_sketchy_dataset/sketches'
    noise_dir = '/home/jefan/full_sketchy_dataset/noise'

    if not os.path.exists(noise_dir):
        os.makedirs(noise_dir)

    photo_class_dirs = os.listdir(photo_dir)
    for class_dir in photo_class_dirs:
        noise_class_path = os.path.join(noise_dir, class_dir)
        if not os.path.exists(noise_class_path):
            os.makedirs(noise_class_path)


    for class_dir in photo_class_dirs:
        class_files = os.listdir(os.path.join(photo_dir, class_dir))

        for class_file in class_files:
            sketch = random_sketch(args.imsize, args.n_strokes)
            save_sketch(sketch, os.path.join(noise_dir, class_dir, class_file))
