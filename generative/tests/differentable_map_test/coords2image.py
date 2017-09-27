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


def coords_to_sketch(endpoints, out_path, use_cuda=False):
    endpoints[:, 0] = endpoints[:, 0] / 640 * 256
    endpoints[:, 1] = endpoints[:, 1] / 480 * 256

    renderer = SketchRenderNet(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], 
                               imsize=256, fuzz=0.3, smoothness=8, use_cuda=use_cuda)

    sketch = renderer()

    # postprocess to make this like a real image
    sketch_min = torch.min(sketch)
    sketch_max = torch.max(sketch)
    sketch = (sketch - sketch_min) / (sketch_max - sketch_min)
    sketch = torch.cat((sketch, sketch, sketch), dim=1)

    sketch_np = sketch[0].cpu().data.numpy() * 255
    sketch_np = np.rollaxis(sketch_np, 0, 3)

    # I am rounding here, which I need to remember to 
    # also do in the wiggle test to make things congruent.
    sketch_np = np.round(sketch_np, 0).astype(np.uint8)
    
    # save image to random path.
    im = Image.fromarray(sketch_np)
    im.save(out_path)


def csv_to_sketch(csv_path, out_folder, use_cuda=False):
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
    print('- Found {} Sketches in CSV File.'.format(len(filenames)))
    for i, filename in enumerate(filenames):
        print('- Rendering sketch [{}/{}]'.format(i + 1, len(filenames)))
        cur_data = data[data[:, -1] == filename]
        cur_data = cur_data[:, 1:4]
        cur_data = cur_data.astype(np.float64)

        out_path = os.path.join(out_folder, filename)
        coords_to_sketch(cur_data, out_path, use_cuda=use_cuda)


if __name__ == "__main__":
    import os
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_folder', type=str)
    # /home/jefan/pix2svg/preprocessing/stroke_dataframes/*.csv
    parser.add_argument('out_folder', type=str)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and args.cuda

    csv_files = glob(os.path.join(args.csv_folder, '*.csv'))
    print('Found {} CSV files.'.format(len(csv_files)))

    for i, csv_path in enumerate(csv_files):
        print('\nProcessing CSV file [{}/{}]: {}'.format(
              i + 1, len(csv_files), csv_path))
        class_name = os.path.splitext(os.path.basename(csv_path))[0]
        folder_path = os.path.join(args.out_folder, class_name)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
            print('Created folder: {}'.format(folder_path))
        csv_to_sketch(csv_path, folder_path, use_cuda=args.cuda)
