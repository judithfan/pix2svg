"""Given a known sketch with its natural photo, we have 
extracted a bunch of fixed coordinates. We can wiggle all 
of them together (using our differentiable line renderer).
We want to make sure it doesn't go anywhere crazy. Small
perturbations are okay and to be expected.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import csv

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms

# these will be used to wiggle.
sys.path.append('../..')
sys.path.append('../distribution_test')
sys.path.append('../multimodal_test')
from linerender import SketchRenderNet
from linerender import BresenhamRenderNet
from distribtest import cosine_similarity
from multimodaltest import load_checkpoint
from precompute_vgg import cnn_predict


photo_preprocessing = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])


def gen_endpoints_from_csv(photo_name, sketch_id):
    csv_path = './data/stroke_dataframe.csv'
    sketch_points = []
    with open(csv_path, 'rb') as fp:
        reader = csv.reader(fp)
        for row in reader:
            if row[-1] == photo_name and int(row[-2]) == int(sketch_id):
                # we are going to ignore pen type (lifting for now)
                sketch_points.append([float(row[1]), float(row[2]), int(row[3])])

    sketch_points = np.array(sketch_points)
    return sketch_points


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='path to the trained model file')
    parser.add_argument('out_folder', type=str,
                        help='where to save sketch')
    parser.add_argument('n_wiggle', type=int, help='number of segments to wiggle (from the end)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    net = load_checkpoint(args.model_path, use_cuda=args.cuda)
    cnn = models.vgg19(pretrained=True)
    cnn.eval()
    net.eval()

    if args.cuda:
        cnn = cnn.cuda()
        net = net.cuda()

    for p in cnn.parameters():
        p.requires_grad = False

    for p in net.parameters():
        p.requires_grad = False

    # TODO: make this not hardcoded.
    photo_path = './data/n02691156_10168.jpg'

    # convert to torch object
    photo = Image.open(photo_path)
    photo = photo.convert('RGB')
    photo = photo_preprocessing(photo).unsqueeze(0)
    photo = Variable(photo, volatile=True)
    if args.cuda:
        photo = photo.cuda()
    photo = cnn_predict(photo, cnn)
    photo = net.photo_adaptor(photo)

    # here we are going to rip photo from its tape and recast it as a variable
    photo = Variable(photo.data)

    # HACK: 0-indexing for sketch_id inside CSV but 1-indexing for sketch_id in filename
    photo_csv_name = os.path.splitext(os.path.basename(photo_path))[0]
    sketch_endpoints = gen_endpoints_from_csv(photo_csv_name, 2)  # HARDCODED ID
    # HACK: coordinates are current in 640 by 480; reshape to 256
    #       AKA: transforms.Scale(256)
    sketch_endpoints[:, 0] = sketch_endpoints[:, 0] / 640 * 256
    sketch_endpoints[:, 1] = sketch_endpoints[:, 1] / 480 * 256 

    renderer = SketchRenderNet(sketch_endpoints[:, 0], sketch_endpoints[:, 1], 
                               sketch_endpoints[:, 2], imsize=256, fuzz=0.3,
                               n_params=args.n_wiggle, use_cuda=args.cuda)
    optimizer = optim.Adam(renderer.parameters(), lr=args.lr)
    if args.cuda:
        renderer = renderer.cuda()


    def train(epoch):
        renderer.train()
        optimizer.zero_grad()
        sketch = renderer()

        # HACK: manually center crop to 224 by 224 from 256 by 256
        #       AKA transforms.CenterCrop(224)
        sketch = sketch[:, :, 16:240, 16:240]
        # HACK: normalize sketch to 0 --> 1 (this is like ToTensor)
        #       AKA transforms.ToTensor()
        sketch_min = torch.min(sketch).expand_as(sketch)
        sketch_max = torch.max(sketch).expand_as(sketch)
        sketch = (sketch - sketch_min) / (sketch_max - sketch_min)
        # HACK: given sketch 3 channels: RGB
        sketch = torch.cat((sketch, sketch, sketch), dim=1)
        # HACK: manually normalize each dimension
        #       AKA transforms.Normalize([0.485, 0.456, 0.406],
        #                                [0.229, 0.224, 0.225])
        sketch[:, 0] = (sketch[:, 0] - 0.485) / 0.229
        sketch[:, 1] = (sketch[:, 1] - 0.456) / 0.224
        sketch[:, 2] = (sketch[:, 2] - 0.406) / 0.225

        sketch = cnn_predict(sketch, cnn)
        sketch = net.sketch_adaptor(sketch)
        
        # we want to optimize 1 - cosine since all the best images were
        # near 1.0. See quip.
        loss = 1 - cosine_similarity(photo, sketch, dim=1)
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            print('Train Epoch: {} \tCosine Distance: {:.6f}'.format(epoch, loss.data[0]))


    for i in range(args.epochs):
        train(i)

    parameters = list(renderer.parameters())
    x_parameters = parameters[0].cpu().data.numpy()
    y_parameters = parameters[1].cpu().data.numpy()
    pen_parameters = renderer.pen_list

    # TODO: BresenhamRenderNet flips x and y; fix this.
    tracer = BresenhamRenderNet(y_parameters, x_parameters,
                                pen_list=pen_parameters, imsize=256)
    sketch = tracer.forward()
    # Do some of preprocessing hacks to make it look lke a real image
    sketch_min = torch.min(sketch)
    sketch_max = torch.max(sketch)
    sketch = (sketch - sketch_min) / (sketch_max - sketch_min)
    sketch = (1 - sketch) * 255
    sketch = torch.cat((sketch, sketch, sketch), dim=1)

    # convert to numpy
    sketch = sketch[0].data.numpy()
    sketch = np.rollaxis(sketch, 0, 3)
    sketch = np.round(sketch, 0).astype(np.uint8)

    # visualize and save
    sketch = Image.fromarray(sketch)
    sketch.save(os.path.join(args.out_folder, 'output.png'))
