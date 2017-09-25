"""For every csv file of coordinates, return a differentiable 
image with a fixed fuzz parameter. 
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import csv
import torch
import numpy as np
from PIL import Image

sys.path.append('../..')
from linerender import SketchRenderNet


def coords_to_sketch(endpoints, out_path):
    endpoints[:, 0] = endpoints[:, 0] / 640 * 256
    endpoints[:, 1] = endpoints[:, 1] / 480 * 256

    renderer = SketchRenderNet(endpoints[:, 0], endpoints[:, 1], 
                               endpoints[:, 2], imsize=256, fuzz=0.0001)

    sketch = renderer()

    # postprocess to make this like a real image
    sketch_min = torch.min(sketch)
    sketch_max = torch.max(sketch)
    sketch = (sketch - sketch_min) / (sketch_max - sketch_min)
    sketch = torch.cat((sketch, sketch, sketch), dim=1)

    sketch_np = sketch[0].data.numpy() * 255
    sketch_np = np.rollaxis(sketch_np, 0, 3)
    sketch_np = np.round(sketch_np, 0).astype(np.uint8)
    
    # save image to random path.
    im = Image.fromarray(sketch_np)
    im.save(out_path)


def csv_to_sketch(csv_path, out_folder):
    """A single csv file is for a number of sketches in a particular 
    class. We should split it up per sketch and save each to a folder
    similar to full_sketch_dataset.
    """
    data = []
    with open(csv_path, 'rb') as fp:
        reader = csv.reader(fp)
        for row in reader:
            data.append(row)

    data = np.array(data)
    filenames = np.unique(data[1:, -1])

    # loop through each sketch by filename
    # grab all the coordinates and generate a sketch
    for filename in filenames:
        cur_data = data[data[:, -1] == filename]
        cur_data = cur_data[:, 1:4]
        cur_data = cur_data.astype(np.float64)

        out_path = os.path.join(out_folder, filename)
        coords_to_sketch(cur_data, out_path)


if __name__ == "__main__":
    import os
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_folder', type=str)
    parser.add_argument('out_folder', type=str)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and args.cuda

    csv_files = glob(os.path.append(args.csv_folder, '*.csv'))

    for csv_path in csv_files:
        class_name = os.path.basename(csv_path)
        folder_path = os.path.join(args.out_folder, class_name)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        csv_to_sketch(csv_path, folder_path)
