from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

import sys; sys.path.append('../..')
from linerender import BresenhamRenderNet
from beamsearch import SemanticBeamSearch
from beamsearch import semantic_sketch_loss
from beamsearch import ALLOWABLE_EMBEDDING_NETS
from beamsearch import ALLOWABLE_DISTANCE_FNS


def save_sketch_to_file(x_paths, y_paths, epoch, outpath='./'):
    # rerender the paths using non-differentiable but pretty renderer
    renderer = BresenhamRenderNet(x_paths, y_paths, imsize=224, linewidth=5)
    sketch = renderer.forward()
    sketch = sketch.int()
    sketch = torch.cat((sketch, sketch, sketch), dim=1)
    sketch = 1 - sketch
    sketch = sketch * 255
    sketch_np = sketch.numpy()[0]
    sketch_np = np.rollaxis(sketch_np, 0, 3)
    sketch_np = sketch_np.astype('uint8')
    im = Image.fromarray(sketch_np)
    im.save(os.path.join(outpath, 'sketch_{}.png'.format(epoch)))


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description="generate sketches")
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--distance', type=str, default='cosine')
    parser.add_argument('--embedder', type=str, default='vgg19')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('folder', type=str)
    parser.add_argument('outpath', type=str)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.distance in ALLOWABLE_DISTANCE_FNS
    assert args.embedder in ALLOWABLE_EMBEDDING_NETS

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

    explorer = SemanticBeamSearch(112, 112, 224, beam_width=2, n_samples=args.n_samples,
                                  n_iters=args.n_epochs, stdev=30, fuzz=0.1,
                                  embedding_net=args.embedder, embedding_layer=args.layer,
                                  use_cuda=args.cuda, distance_fn=args.distance, verbose=True)

    natural_emb = explorer.embedding_net(natural)
    distractor_embs = explorer.embedding_net(distractors)

    for i in range(args.n_epochs):
        sketch = explorer.train(i, natural_emb, distractor_items=distractor_embs)

        x_paths, y_paths = explorer.gen_paths()
        save_sketch_to_file(x_paths[:i+2], y_paths[:i+2], i, outpath=args.outpath)

    gt_sketch = Image.open(os.path.join(args.folder, 'sketch.png'))
    gt_sketch = gt_sketch.convert('RGB')
    gt_sketch = preprocessing(gt_sketch).unsqueeze(0)
    if args.cuda:
        gt_sketch = gt_sketch.cuda()
    gt_sketch = Variable(gt_sketch, volatile=True)
    if args.cuda:
        gt_sketch = gt_sketch.cuda()

    sketch_emb = explorer.preprocess_sketches(sketch.unsqueeze(0))
    gt_sketch_emb = explorer.embedding_net(gt_sketch)

    pred_dist = semantic_sketch_loss(natural_emb, sketch_emb, distractor_embs,
                                     distance_fn=args.distance, use_cuda=args.cuda)
    gt_dist = semantic_sketch_loss(natural_emb, gt_sketch_emb, distractor_embs,
                                   distance_fn=args.distance, use_cuda=args.cuda)

    print("True Sketch & Natural Image Loss: {}".format(gt_dist.data[0]))
    print("Generated Sketch & Natural Image Loss: {}".format(pred_dist.data[0]))
