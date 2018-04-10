from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import json
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.autograd import Variable
from dataset import ExhaustiveDataset
from train import load_checkpoint


def photo_uname(path):
    path = os.path.splitext(os.path.basename(path))[0]
    return path


def sketch_uname(path):
    path = '_'.join(os.path.splitext(os.path.basename(path))[0].split('_')[1:])
    path = path.split('-')[-1]
    path = path.replace('_trial', '')
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    
    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if model.cuda:
        model.cuda()

    dataset = ExhaustiveDataset(layer='conv42')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        
    dist_jsons = defaultdict(lambda: {})
    pbar = tqdm(total=len(loader))
    for batch_idx, (sketch, sketch_object, sketch_context, sketch_path) in enumerate(loader):
        sketch_name = sketch_uname(sketch_path[0])
        sketch = Variable(sketch, volatile=True)
        if args.cuda:
            sketch = sketch.cuda()
        photo_generator = dataset.gen_photos()
        for photo, photo_object, photo_path in photo_generator():
            photo_name = photo_uname(photo_path)
            photo = Variable(photo, volatile=True)
            batch_size = len(sketch)
            if args.cuda:
                photo = photo.cuda()
            pred = model(photo, sketch).squeeze(1).cpu().data[0]
            dist_jsons[photo_name][sketch_name] = float(pred)
        pbar.update()
    pbar.close()

    with open('./dump.json', 'w') as fp:
        json.dump(dist_jsons, fp)
