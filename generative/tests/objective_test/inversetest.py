"""Motivated by the weird finding that the best sketches have the 
highest loss and the small tiny bad sketches have the lowest loss, 
let's use conv_4_2 and go in the other direction.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

import sys; sys.path.append('../..')
from linerender import BresenhamRenderNet
from beamsearch import sample_endpoint_gaussian2d
from distribtest import SingleLayerLossTest


beam_width = 2
n_iters = 40
n_samples = 2000
use_cuda = torch.cuda.is_available()
imsize = 224
stdev = 10
x0, y0 = imsize // 2, imsize // 2

# for loss we will use the conv_4_2 layer
evaluator = SingleLayerLossTest('conv_4_2', 'euclidean', use_cuda=use_cuda)

photo = Image.open('/home/jefan/full_sketchy_dataset/photos/airplane/n02691156_10168.jpg')
photo = photo.convert('RGB')
print('loaded photo image')

preprocessing = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(imsize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

photo = Variable(preprocessing(photo).unsqueeze(0))
if use_cuda:
    photo = photo.cuda()

x_beam_queue = np.ones(beam_width) * x0
y_beam_queue = np.ones(beam_width) * y0

x_beam_paths = np.zeros((beam_width, n_iters + 1))
y_beam_paths = np.zeros((beam_width, n_iters + 1))

x_beam_paths[:, 0] = x0
y_beam_paths[:, 0] = y0


def train(epoch):
    global x_beam_queue
    global y_beam_queue
    global x_beam_paths
    global y_beam_paths
    print('epoch [{}/{}]'.format(iter + 1, n_iters))

    for b in range(beam_width):
        print('- beam [{}/{}]'.format(b + 1, beam_width))
        # sample endpoints
        samples = sample_endpoint_gaussian2d(x_beam_queue[b], y_beam_queue[b],
                                             std=stdev, size=n_samples,
                                             min_x=0, max_x=imsize,
                                             min_y=0, max_y=imsize)
        x_samples, y_samples = samples[:, 0], samples[:, 1]
        print('-- sampled {} points'.format(n_samples))
        losses = torch.zeros((n_samples))

        # for each sample & render image
        for i in range(n_samples):
            x_list = copy.deepcopy(x_beam_paths[b])
            y_list = copy.deepcopy(y_beam_paths[b])
            x_list[epoch + 1] = x_samples[i]
            y_list[epoch + 1] = y_samples[i]
            renderer = BresenhamRenderNet(x_list[:epoch + 2], y_list[:epoch + 2],
                                          imsize=imsize, linewidth=5)
            sketch = renderer.forward()
            sketch = torch.cat((sketch, sketch, sketch), dim=1)
            sketch = Variable(sketch, volatile=True)
            if use_cuda:
                sketch = sketch.cuda()
            loss = evaluator.loss(photo, sketch) 
            # make inverse
            loss = loss * -1.0
            losses[i] = float(loss.cpu().data.numpy()[0])
            if (i + 1) % 25 == 0:
                print('--- calc loss for [{}/{}] samples'.format(i + 1, n_samples))

        if b == 0:
            beam_losses = losses.numpy()
            x_beam_samples = x_samples
            y_beam_samples = y_samples
        else:
            beam_losses = np.concatenate((beam_losses, losses.numpy()))
            x_beam_samples = np.concatenate((x_beam_samples, x_samples))
            y_beam_samples = np.concatenate((y_beam_samples, y_samples))

    top_ii = np.argsort(beam_losses)[:beam_width]
    _x_beam_paths = copy.deepcopy(x_beam_paths)
    _y_beam_paths = copy.deepcopy(y_beam_paths)

    for b in range(beam_width):
        parent = top_ii[b] // n_samples
        _x_beam_paths[b][epoch + 1] = x_beam_samples[top_ii[b]]
        _y_beam_paths[b][epoch + 1] = y_beam_samples[top_ii[b]]

    x_beam_paths = _x_beam_paths
    y_beam_paths = _y_beam_paths
    x_beam_queue = np.array([x_beam_samples[top_ii[b]] for b in range(beam_width)])
    y_beam_queue = np.array([y_beam_samples[top_ii[b]] for b in range(beam_width)])

    best_ii = top_ii[0] // n_samples
    x_list = x_beam_paths[best_ii][:epoch + 2]
    y_list = y_beam_paths[best_ii][:epoch + 2]
    print('- updated global beam variables...')

    top_renderer = BresenhamRenderNet(x_list, y_list, imsize=imsize, linewidth=5)
    top_sketch = top_renderer.forward()
    print('- generated top sketch | loss: {}'.format(beam_losses[best_ii]))

    return top_sketch


def save_sketch(sketch, epoch, outfolder='./'):
    sketch = sketch.int()
    sketch = torch.cat((sketch, sketch, sketch), dim=1)
    sketch = (1 - sketch) * 255
    sketch_np = np.rollaxis(sketch.numpy()[0], 0, 3).astype('uint8')
    im = Image.fromarray(sketch_np)
    im.save(os.path.join(outfolder, 'sketch_{}.png'.format(epoch)))


print('training beam model...')
for iter in range(n_iters):
    sketch = train(iter)
    save_sketch(sketch, iter, outfolder='./outputs_inverse')

print('saving final sketch...')
save_sketch(sketch, 'final', outfolder='./outputs_inverse')
