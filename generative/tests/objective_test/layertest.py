from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

import sys; sys.path.append('../..')
from beamsearch import SemanticBeamSearch
from beamsearch import semantic_sketch_loss


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description="generate sketches")
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('folder', type=str)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    natural = Image.open(os.path.join(args.folder, 'natural.png'))
    natural = natural.convert('RGB')
    print('loaded natural image')

    distractors = []
    distractors_folder = os.path.join(args.folder, 'distractors')
    bad_files = ['.DS_Store']
    distractors_files = [p for p in os.listdir(distractors_folder)
                         if p not in bad_files]

    for i in distractors_files:
        distractor_path = os.path.join(distractors_folder, i)
        distractor = Image.open(distractor_path)
        distractor = distractor.convert('RGB')
        distractors.append(distractor)
    print('loaded distractor images')

    preprocessing = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    natural = preprocessing(natural).unsqueeze(0)
    distractors = torch.cat([preprocessing(image).unsqueeze(0) for image in distractors])
    if args.cuda:
        natural, distractors = natural.cuda(), distractors.cuda()
    natural, distractors = Variable(natural), Variable(distractors)

    explorer = SemanticBeamSearch(112, 112, 224, beam_width=4, n_samples=100,
                                  n_iters=20, stdev=20, fuzz=0.1,
                                  embedding_layer=args.layer, use_cuda=args.cuda,
                                  verbose=True)

    natural_emb = explorer.embedding_net(natural)
    distractor_embs = explorer.embedding_net(distractors)

    for i in range(20):
        sketch = explorer.train(i, natural_emb, distractor_items=distractor_embs)

    im = Image.fromarray(sketch)
    im.save('./sketch.png')

    gt_sketch = Image.open(os.path.join(args.folder, 'natural.png'))
    gt_sketch = gt_sketch.convert('RGB')
    gt_sketch = preprocessing(gt_sketch).unsqueeze(0)
    if args.cuda:
        gt_sketch = gt_sketch.cuda()
    gt_sketch = Variable(gt_sketch)
    sketch_emb = explorer.preprocess_sketches(sketch.unsqueeze(0))

    pred_dist = semantic_sketch_loss(natural_emb, sketch_emb, distractor_embs)
    gt_dist = semantic_sketch_loss(natural_emb, gt_sketch_emb, distractor_embs)

    print("True Sketch & Natural Image Loss: {}".format(gt_dist.data[0]))
    print("Generated Sketch & Natural Image Loss: {}".format(pred_dist.data[0]))
