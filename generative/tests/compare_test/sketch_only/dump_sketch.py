from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import json
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from dataset_sketch import ExhaustiveSketchDataset
from train_sketch import load_checkpoint


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

    dataset = ExhaustiveSketchDataset(layer='conv42', split='test')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    photo_ordering = dataset.object_order
        
    dist_jsons = defaultdict(lambda: {})
    pbar = tqdm(total=len(loader))
    test_sketchpaths = []
    for batch_idx, (sketch, sketch_object, sketch_context, sketch_path) in enumerate(loader):
        sketch_name = sketch_uname(sketch_path[0])
        test_sketchpaths.append(os.path.basename(sketch_path[0]))
        sketch = Variable(sketch, volatile=True)
        if args.cuda:
            sketch = sketch.cuda()
        pred_logits = model(sketch)
        if model.xent_loss:
            pred = F.softmax(pred_logits, dim=1)
        else:
            pred = F.softplus(pred_logits)
            pred = pred / torch.sum(pred, dim=1, keepdim=True)
        pred = pred.cpu().data[0]  # single batch
        for dim, photo_name in enumerate(photo_ordering):
            dist_jsons[photo_name][sketch_name] = float(pred[dim])
        pbar.update()
    pbar.close()

    with open('./dump.json', 'w') as fp:
        json.dump(dist_jsons, fp)

    with open('./dump-paths.json', 'w') as fp:
        json.dump(test_sketchpaths, fp)

