from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

from linerender import RenderNet
from vggutils import (VGG19Split, vgg_convert_to_avg_pool)

ALLOWABLE_SAMPLING_PRIORS = ['gaussian', 'angle']
ALLOWABLE_ACTIONS = ['draw', 'move']


def mean_pixel_sketch_loss(natural_image, sketch_images, distractor_images=None,
                           segment_cost=0.0):
    losses = pixel_sketch_loss(natural_image, sketch_images,
                               distractor_images=distractor_images,
                               segment_cost=segment_cost)
    return torch.mean(losses)


def pixel_sketch_loss(natural_image, sketch_images, distractor_images=None,
                      segment_cost=0.0):
    """Calculate L2 distance between natural image and sketch
    image + normalization using distractor images.

    :param natural_image: PyTorch Tensor 1xCxHxW
    :param sketch_images: PyTorch Tensor SxCxHxW
    :param distractor_images: PyTorch Tensor DxCxHxW (default None)
    :param segment_cost: cost of adding this segment (default 0.0)
    """
    n_sketches = sketch_images.size()[0]
    # flatten images into a vector
    natural_image = natural_image.view(1, -1)
    sketch_images = sketch_images.view(n_sketches, -1)

    # distance between natural image and each sketch
    loss = torch.sum(torch.pow(natural_image - sketch_images, 2), dim=1)
    loss = torch.unsqueeze(loss, dim=0)

    if distractor_images is not None:
        n_distractors = distractor_images.size()[0]
        distractor_images = distractor_images.view(n_distractors, -1)

        # distance between distractor image and each sketch
        distraction_dists = Variable(torch.zeros((n_distractors, n_sketches)))
        for i in range(n_distractors):
            distraction_dists[i] = torch.sum(torch.pow(distractor_images[i]
                                                       - sketch_images, 2), dim=1)
        all_dists = torch.cat((loss, distraction_dists))
        norm = torch.norm(all_dists, p=2, dim=0)
        loss = loss / norm

    return loss + segment_cost


def mean_semantic_sketch_loss(natural_emb, sketch_embs, distractor_embs=None, distance_fn = 'cosine', 
                              segment_cost=0.0):
    losses = semantic_sketch_loss(natural_emb, sketch_embs,
                                  distractor_embs=distractor_embs,
                                  distance_fn = distance_fn,
                                  segment_cost=segment_cost)
    return torch.mean(losses)


def semantic_sketch_loss(natural_emb, sketch_embs, distractor_embs=None, distance_fn = 'cosine', 
                         segment_cost=0.0):
    """Calculate distance between natural image and sketch image
    where the distance is normalized by distractor images.

    :param natural_emb: tuple of PyTorch Variable
    :param sketch_embs: tuple of PyTorch Variables
    :param distractor_embs: tuple of PyTorch Variables (default None)
    :param distance_fn: string defining the type of distance function to use (default cosine)
    :param segment_cost: cost of adding this segment (default 0)
    """
    n_sketches = sketch_embs[0].size()[0]
    n_features = len(natural_emb)

    loss = Variable(torch.FloatTensor(n_sketches).zero_(), requires_grad=True)

    for f in range(n_features):
        if distance_fn in ['cosine','correlation']:
            costs = F.cosine_similarity(natural_emb[f], sketch_embs[f], dim=1)    
        elif distance_fn in ['L_1','l1','manhattan']:
            costs = F.pairwise_distance(natural_emb[f], sketch_embs[f], p=1)            
        elif distance_fn in ['L_2','l2','euclidean']:
            costs = F.pairwise_distance(natural_emb[f], sketch_embs[f], p=2)            
        loss = torch.add(loss, costs)

    if distractor_embs is not None:
        n_distractors = distractor_embs[0].size()[0]
        distraction_dists = Variable(torch.zeros((n_distractors, n_sketches)))
        for j in range(n_distractors):
            for f in range(n_features):
                distraction_emb = torch.unsqueeze(distractor_embs[f][j], dim=0)
                if distance_fn in ['cosine','correlation']:
                    costs = F.cosine_similarity(distraction_emb, sketch_embs[f], dim=1)    
                elif distance_fn in ['L_1','l1','manhattan']:
                    costs = F.pairwise_distance(distraction_emb, sketch_embs[f], p=1)            
                elif distance_fn in ['L_2','l2','euclidean']:
                    costs = F.pairwise_distance(distraction_emb, sketch_embs[f], p=2)             
                distraction_dists[j] = torch.add(distraction_dists[j], costs)

        all_dists = torch.cat((torch.unsqueeze(loss, dim=0), distraction_dists))
        norm = torch.norm(all_dists, p=2, dim=0)
        loss = loss / norm

    return loss + segment_cost


def gen_action():
    """Sample an action from an action space of 'draw' or 'move'
    For now, we simply always return 'draw'.
    """
    return 'draw'


def sample_endpoint_gaussian2d(x_s, y_s, std=10, size=1, min_x=0, max_x=224, min_y=0, max_y=224):
    """Sample a coordinate (x_e, y_e) from a 2D gaussian with set deviation.
    The idea is to bias coordinates closer to the start to make smaller strokes.

    :param x_s: starting x coordinate
    :param y_s: starting y coordinate
    :param std: default 10 - controls stroke size
    :param size: default 1 - number of sample
    :param min_x: default 0
    :param max_x: default 224
    :param min_y: default 0
    :param max_y: default 224
    :return samples: 2d array of x_e and y_e coordinates
    """
    mean = np.array([x_s, y_s])
    cov = np.eye(2) * std**2
    samples = np.random.multivariate_normal(mean, cov, size=size)
    # cut off boundaries (this happens if std is too big)
    samples[:, 0][samples[:, 0] < min_x] = min_x
    samples[:, 0][samples[:, 0] > max_x] = max_x
    samples[:, 1][samples[:, 1] < min_y] = min_y
    samples[:, 1][samples[:, 1] > max_y] = max_y
    return np.round(samples, 0).astype(int)


def sample_endpoint_angle(x_s, y_s, x_l, y_l, std=10, angle_std=60, size=1,
                          min_x=0, max_x=224, min_y=0, max_y=224):
    """Sample a coordinate (x_e, y_e) by sampling an angle and a distance
    from two separate 1d gaussians. The idea here is to prevent sampling
    lines that go back and forth

    :param x_s: starting x coordinate
    :param y_s: starting y coordinate
    :param x_l: last x coordinate
    :param y_l: last y coordinate
    :param std: default 10 - controls stroke size
    :param size: default 1 - number of sample
    :param min_x: default 0
    :param max_x: default 224
    :param min_y: default 0
    :param max_y: default 224
    :return samples: 2d array of x_e and y_e coordinates
    """
    d = math.sqrt((x_s - x_l)**2 + (y_s - y_l)**2)
    init_angle = np.rad2deg(np.arccos((x_s - x_l) / d))

    angles = np.clip(np.random.normal(loc=init_angle, scale=angle_std, size=size),
                    -180 + init_angle, 180 + init_angle)
    lengths = np.random.normal(loc=d, scale=std, size=size)

    x_samples = lengths * np.cos(np.deg2rad(angles)) + x_s
    y_samples = lengths * np.sin(np.deg2rad(angles)) + y_s
    samples = np.vstack((x_samples, y_samples)).T

    # cut off boundaries (this happens if std is too big)
    samples[:, 0][samples[:, 0] < min_x] = min_x
    samples[:, 0][samples[:, 0] > max_x] = max_x
    samples[:, 1][samples[:, 1] < min_y] = min_y
    samples[:, 1][samples[:, 1] > max_y] = max_y
    return np.round(samples, 0).astype(int)


def pytorch_batch_exec(fn, objects, batch_size, out_dim=1):
    """ Batch execution on Pytorch variables using Pytorch function.
    The intended purpose is to save memory.

    :param fn: function must take a PyTorch variable of sketches as input
    :param objects: PyTorch Variable containing data
    :param batch_size: number to process at a time
    :param out_dim: fn() returns how many outputs?
    :return: [Variable, ...] list of size out_dim
    """
    num_objects = objects.size()[0]  # number of images total
    num_reads = int(math.floor(num_objects / batch_size))  # number of passes needed
    num_processed = 0  # track the number of batches processed
    out_arr = [[] for o in range(out_dim)]

    for i in range(num_reads):
        objects_batch = objects[
            num_processed:num_processed+batch_size,
        ]
        out_batch = fn(objects_batch)
        for o in range(out_dim):
            out_arr[o].append(out_batch[o])
        num_processed += batch_size

    # process remaining images
    if num_objects - num_processed > 0:
        objects_batch = objects[
            num_processed:num_objects,
        ]
        out_batch = fn(objects_batch)
        for o in range(out_dim):
            out_arr[o].append(out_batch[o])

    # stack all of them together
    return [np.vstack(arr) for arr in out_arr]


def load_vgg19(max_to_avg_pool=False, vgg_layer_index=-1):
    vgg19 = models.vgg19(pretrained=True)
    if max_to_avg_pool:  # replace MaxPool2D with AveragePool2D
        vgg19 = vgg_convert_to_avg_pool(vgg19)
    vgg19 = VGG19Split(vgg19, vgg_layer_index)
    vgg19.eval()  # freeze dropout

    # freeze each parameter
    for p in vgg19.parameters():
        p.requires_grad = False

    return vgg19


def train(natural_image, distractor_images, **kwargs):
    """ Performs beam search training to generate a sketch that is
    semantically similar to a natural image provided some context.

    :param natural_image: PIL array WxHxC
    :param distractor_images: list of PIL array [WxHxC]
    :return: [x_coords, y_coords, actions] that led to a good sketch.
    """
    # load the deep net
    vgg_layer_index = kwargs.get('vgg_layer_index', -1)
    vgg19 = load_vgg(max_to_avg_pool=kwargs.get('max_to_avg_pool', False),
                     vgg_layer_index=vgg_layer_index)


    preprocessing = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    # grab embeddings for the natural & distractor images
    natural = Variable(preprocessing(natural_image).unsqueeze(0))
    distractors = Variable(torch.cat([preprocessing(image).unsqueeze(0)
                                      for image in distractor_images]))
    natural_emb = vgg19(natural)
    distractor_embs = vgg19(distractors)

    # constants for training
    n_features = 8 if vgg_layer_index == -1 else 1
    n_distractors = distractor_images.shape[0]
    x_init, y_init = 112, 112

    # variables for early-stopping
    best_loss = -np.inf
    best_loss_iter = 0
    best_loss_beam = 0
    best_sketch = None
    segment_cost = 0.0
    patience = kwargs.get('patience', 5)
    beam_width = kwargs.get('beam_width', 2)
    n_iters = kwargs.get('n_iters', 100)
    sampling_prior = kwargs.get('sampling_prior', 'gaussian')
    n_samples = kwargs.get('n_samples', 5)

    # each queue holds future paths to try
    # for the first iteration, use beam_width repetitions of the init point
    # - a little inefficient but simple!
    x_beam_queue = np.ones(beam_width) * x_init
    y_beam_queue = np.ones(beam_width) * y_init

    # store full path for top particles
    x_beam_paths = np.zeros((beam_width, n_iters + 1))
    y_beam_paths = np.zeros((beam_width, n_iters + 1))
    action_beam_paths = [[] for b in range(beam_width)]
    x_beam_paths[:, 0] = x_init
    y_beam_paths[:, 0] = y_init

    # store the current sketch per beam
    beam_sketches = Variable(torch.zeros((beam_width, 3, 224, 224)))

    for iter in range(n_iters):
        print('ITER (%d/%d) - BEAM QUEUE LENGTH: %d' %
              (iter + 1, n_iters, beam_width))

        # beam search across a fixed width
        for b in range(beam_width):
            # sample new coordinates in 2 ways
            if args.sampling_prior == 'gaussian' or iter == 0:  # 1st iter has to be gaussian
                std = kwargs.get('std', 15)
                coord_samples = sample_endpoint_gaussian2d(x_beam_queue[b], y_beam_queue[b],
                                                           std=std, size=n_samples,
                                                           min_x=0, max_x=224, min_y=0, max_y=224)
            elif args.sampling_prior == 'angle':
                std = kwargs.get('std', 15)
                angle_std = kwargs.get('angle_std', 15)
                coord_samples = sample_endpoint_angle(x_beam_queue[b], y_beam_queue[b],
                                                      # estimate local context by the last 3 drawn points!
                                                      np.mean(x_beam_paths[b, -3:]),
                                                      np.mean(y_beam_paths[b, -3:]),
                                                      std=std, angle_std=angle_std, size=n_samples,
                                                      min_x=0, max_x=224, min_y=0, max_y=224)

            x_samples, y_samples = coord_samples[:, 0], coord_samples[:, 1]
            print('- generated %d samples' % args.n_samples)

            # sample an action e.g. draw vs move pen
            action_sample = gen_action()  # TODO: make probabilistic
            print('- sampled \'%s\' action' % action_sample)

            # the cost of drawing a new line increases over time
            if action_sample == 'draw':
                segment_cost += kwargs.get('segment_cost', 0.)

            # for each sample, render the image as a matrix
            sketches = Variable(torch.zeros((n_samples, 3, 224, 224)))
            sketches_raw = Variable(torch.zeros((n_samples, 3, 224, 224)))

            for i in range(n_samples):
                action_path = action_beam_paths[b] + [action_sample]
                renderer = RenderNet(x_beam_paths[b][-1], y_beam_paths[b][-1],
                                     x_samples[i], y_samples[i], imsize=224)
                sketch = renderer()
                if iter > 0:  # first iter is just all 0's
                    sketch += beam_sketches[b]
                    sketch = sketch.clamp(0, 1)
                sketches_raw[i] = sketch  #  save raw sketch

                # manually apply preprocessing
                sketch[0] = (sketch[0] - 0.485) / 0.229
                sketch[1] = (sketch[1] - 0.456) / 0.224
                sketch[2] = (sketch[2] - 0.406) / 0.225
                sketches[i] = sketch

            # compute loss functions
            sketch_embs = vgg19(sketches)
            losses = semantic_sketch_loss(natural_emb, sketch_embs, distractor_embs, distance_fn='cosine',
                                          segment_cost=segment_cost)
            print('- computed losses')

            if b == 0:
                beam_losses = losses.data.numpy()
                x_beam_samples = x_samples
                y_beam_samples = y_samples
                action_beam_samples = np.array([action_sample])
            else:
                beam_losses = np.concatenate((beam_losses, losses.data.numpy()))
                x_beam_samples = np.concatenate((x_beam_samples, x_samples))
                y_beam_samples = np.concatenate((y_beam_samples, y_samples))
                action_beam_samples = np.append(action_beam_samples, action_sample)

            print('Finished beam particle %d\n' % (b + 1))

        # keep the top N particles
        top_indexes = np.argsort(beam_losses)[::-1][:beam_width]

        # these will hold updated beam paths
        for b in range(beam_width):
            # which beam it originated from
            beam_parent = int(np.floor(top_indexes[b] / n_samples))
            x_beam_paths[b, :] = np.append(x_beam_paths[beam_parent, :],
                                           x_beam_samples[top_indexes[b]])
            y_beam_paths[b, :] = np.append(y_beam_paths[beam_parent, :],
                                           y_beam_samples[top_indexes[b]])
            action_beam_paths[b].append(action_beam_samples[b])

        # update beam queue with top beam_width samples across beams
        x_beam_queue = np.array([x_beam_samples[top_indexes[b]]
                                 for b in range(beam_width)])
        y_beam_queue = np.array([y_beam_samples[top_indexes[b]]
                                 for b in range(beam_width)])

        # update our sketch per beam
        for b in range(beam_width):
            beam_sketch = beam_sketches[b]
            beam_sketch += sketches_raw[top_indexes[b]]
            beam_sketch = beam_sketch.clamp(0, 1)
            beam_sketches[b] = beam_sketch

        top_beam_loss = beam_losses[top_indexes[0]]
         # update patience constants (and possibly break)
        if top_beam_loss >= best_loss:
            best_loss = top_beam_loss
            best_loss_beam = int(np.floor(top_indexes[0] / n_samples))
            best_loss_iter = iter
            best_sketch = beam_sketches[top_indexes[0]]
            print('- new best loss %f. patience reset to %d' % (best_loss, patience))
        else:
            patience -= 1
            print('- no improvement %f. patience lowered to %d' % (best_loss, patience))

        if patience <= 0:  # this is equivalent to "stop" action
            print('- out of patience. quitting...')
            break
        print('')  # new line

    return best_sketch


if __name__ == '__main__':
    import json
    import argparse
    from PIL import Image

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')

    parser = argparse.ArgumentParser(description="generate sketches")
    parser.add_argument('--image_path', type=str, help='path to image file')
    parser.add_argument('--distract_dir', type=str, help='directory to distractor image files')
    parser.add_argument('--sketch_dir', type=str, help='directory to store sketches')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='number of samples per iteration')
    parser.add_argument('--n_iters', type=int, default=20,
                        help='number of iterations')
    parser.add_argument('--std', type=float, default=15.0,
                        help='std for Gaussian when sampling')
    parser.add_argument('--patience', type=int, default=5,
                        help='once the informativity measure stops improving, wait N epochs before quitting')
    parser.add_argument('--beam_width', type=int, default=2,
                        help='number of particles to preserve at each timestep')
    parser.add_argument('--vgg_layer_index', type=int, default=-1,
                        help='index of layer in vgg19 for feature extraction. Pass -1 to use all layers...')
    parser.add_argument('--sampling_prior', type=str, default='gaussian',
                        help='gaussian|angle')
    parser.add_argument('--angle_std', type=int, default=60,
                        help='std for angles when sampling_prior == angle')
    parser.add_argument('--max_to_avg_pool', action='store_true',
                        help='if true, replace MaxPool2D with AvgPool2D in VGG19')
    parser.add_argument('--segment_cost', type=float, default=0.0,
                        help='cost for drawing a single segment')
    args = parser.parse_args()

    assert args.beam_width <= args.n_samples
    assert args.vgg_layer_index >= -1 and args.vgg_layer_index < 45
    assert args.sampling_prior in ALLOWABLE_SAMPLING_PRIORS

    # prep images
    natural = Image.open(args.image_path)
    distractors = []
    for i in os.listdir(args.distract_dir):
        distractor_path = os.path.join(args.distract_dir, i)
        distractor = Image.open(distractor_path)
        distractors.append(distractor)

    train_params = {'n_samples': args.n_samples,
                    'n_iters': args.n_iters,
                    'std': args.std,
                    'angle_std': args.angle_std,
                    'patience': args.patience,
                    'beam_width': args.beam_width,
                    'vgg_layer_index': args.vgg_layer_index,
                    'sampling_prior': args.sampling_prior,
                    'max_to_avg_pool': args.max_to_avg_pool,
                    'segment_cost': args.max_to_avg_pool}
    best_sketch = train(natural, distractors)
    sketch_path = os.path.join(args.sketch_dir, 'sketch.png', **train_params)
    im = Image.fromarray(sketch_path)
    im.save(sketch_path)