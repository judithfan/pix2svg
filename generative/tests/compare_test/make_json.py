from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import json
import torch

from dataset import ExhaustiveDataset
from train import load_checkpoint


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
        
    dist_jsons = []
    pbar = tqdm(total=len(loader))
    for batch_idx, (sketch, sketch_object, sketch_context, sketch_path) in enumerate(train_loader):
        photo_generator = dataset.gen_photos()
        for photo, photo_object, photo_path in photo_generator:
            sketch = Variable(sketch, volatile=True)
            photo = Variable(photo, volatile=True)
            batch_size = len(sketch)
            if args.cuda:
                photo = photo.cuda()
                sketch = sketch.cuda()
            pred = model(photo, sketch).cpu().data
            dist_jsons.append({'sketch': sketch_path,
                               'photo': photo_path,
                               'distance': pred})
        pbar.update()
    pbar.close()

    print('\nWriting distances to file.')
    with open(args.json_path, 'w') as fp:
        json.dump(dist_jsons, fp)
