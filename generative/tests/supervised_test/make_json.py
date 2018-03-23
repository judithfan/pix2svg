from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable

from model import SketchNet
from dataset import SketchPlus32Photos
from train import load_checkpoint 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', type=str, help='path to where to dump json')
    parser.add_argument('--model-path', type=str, default='./trained_models/model_best.pth.tar', 
                        help='where trained models are stored.')
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path)    
    model.eval()
    if args.cuda:
        model.cuda()

    state_dict = (torch.load(args.model_path) if args.cuda else
                  torch.load(args.model_path, map_location=lambda storage, location: storage))
    model.load_state_dict(state_dict)
    loader = torch.utils.data.DataLoader(SketchPlus32Photos(layer=model.layer, return_paths=True), 
                                         batch_size=1)

    dist_jsons = []
    pbar = tqdm(total=len(loader))
    for photos, sketch, photo_paths, sketch_path, label in loader:
        photos = Variable(photos, volatile=True)
        sketch = Variable(sketch, volatile=True)
        label = Variable(label, requires_grad=False)
        batch_size = len(photos)

        if args.cuda:
            photos = photos.cuda()
            sketch = sketch.cuda()
            label = label.cuda()

        log_distance = model(photos, sketch)
        distance = torch.exp(distance)
        distance = distance.cpu().data.numpy().flatten()
        for i in xrange(32):
            output = {u'distance': float(distances[i]),
                      u'render': str(photo_paths[i]),
                      u'sketch': str(sketch_path)}
            dist_jsons.append(output)
        pbar.update()
    pbar.close()

    with open(args.json_path, 'w') as fp:
        json.dump(dist_jsons, fp)
