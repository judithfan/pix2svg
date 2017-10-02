from __future__ import division
import os

import numpy as np
import scipy.stats as stats
import pandas as pd
import json

from PIL import Image
import base64

from generator import alpha_composite_with_color

import torch
from torch.nn import *
from generator import ReferenceGameGenerator
import torchvision.models as models

## path to sketches
path_to_sketches = '/home/jefan/sketchpad_basic_extract/sketch'
## path to 3D renderings
path_to_renderings = '/home/jefan/sketchpad_basic_extract/subordinate_allrotations_6_minified'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract pixel-level distances between sketches and 3D renderings')
    parser.add_argument('--sketch_path', type=str, default='/home/jefan/sketchpad_basic_extract/sketch')
    parser.add_argument('--render_path', type=str, default= '/home/jefan/sketchpad_basic_extract/subordinate_allrotations_6_minified')
    parser.add_argument('--json_path', type=str, default='/home/jefan/pix2svg/generative/tests/reference_games/json/pixel_dist.json')    
    parser.add_argument('--extension', type=str, help='jpg|png')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    
    _generator = ReferenceGameGenerator(args.sketch_path, args.render_path, 
                                                 use_cuda=use_cuda)
    generator = _generator.make_generator() 

    dist_jsons = []
    count = 0
    while True:
        try:  # exhaust the generator to return all pairs
            sketch_path, sketch, render_path, render = generator.next()
        except StopIteration:
            break

        cos = CosineSimilarity(dim=0, eps=1e-6)
        dist = cos(sketch.view(-1),render.view(-1)).cpu().data.numpy()[0]

        dist_json = {'sketch': sketch_path,
                     'render': render_path,
                     'distance': dist}

        dist_jsons.append(dist_json)

        print('Compute Distance [{}/{}].'.format(count + 1, _generator.size))
        count += 1

    print('\nWriting distances to {}.')
    with open(args.json_path, 'w') as fp:
        json.dump(dist_jsons, fp)

