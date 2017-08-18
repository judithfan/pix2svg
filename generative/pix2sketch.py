from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

from linerender import RenderNet
from vggutils import (VGG19Split, build_vgg19_feature_extractor,
                      vgg_convert_to_avg_pool)

ALLOWABLE_SAMPLING_PRIORS = ['gaussian', 'angle']
ALLOWABLE_ACTIONS = ['draw', 'move']


class SketchLoss(nn.Module):
    """Calculate distance between natural image and sketch image
    where the distance is normalized by distractor images.

    :param n_features: number of layer embeddings
    :param n_distractors: number of distractor images
    :param segment_cost: cost of adding this segment
    """

    def __init__(self, n_features, n_distractors, segment_cost=0.0):
        self.segment_cost = segment_cost
        self.n_features = n_features
        self.n_distractors = n_distractors

    def forward(self, natural_emb, sketch_emb, distractor_embs):
        natural_dist = Variable(torch.Tensor([0]))
        for f in range(self.n_features):
            natural_dist += F.cosine_similarity(natural_emb[f], sketch_emb[f])

        distraction_dists = Variable(torch.zeros(self.n_distractors))
        for j in range(self.n_distractors):
            distraction_dist = Variable(torch.Tensor([0]))
            for f in range(self.n_features):
                distraction_dist += F.cosine_similarity(torch.unsqueeze(distractor_embs[f][j], dim=0),
                                                        sketch_emb[f])
            distraction_dists[j] = distraction_dist

        all_dists = torch.cat([natural_dist, distraction_dists])
        loss = natural_dist / torch.norm(all_dists, p=2) + segment_cost

        return loss


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


def load_images_to_torch(paths, preprocessing=None):
    """Load a bunch jpegs/pngs/etc. into a Torch environment
    as a single Torch Variable.
    :param paths: list of str
    :param preprocessing: transforms.Compose([...])
    :return: Torch Variable
    """
    num_paths = len(paths)
    imgs = []
    for i in range(num_paths):
        img = Image.open(paths[i])
        if preprocessing is not None:
            img = preprocessing(img).unsqueeze(0)
        imgs.append(img)
    imgs = torch.cat(imgs)
    return Variable(imgs)


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
    parser.add_argument('--batch_size', type=int, default=100,
                        help='size when running forward pass (memory issues)')
    parser.add_argument('--std', type=float, default=15.0,
                        help='std for Gaussian when sampling')
    parser.add_argument('--patience', type=int, default=5,
                        help='once the informativity measure stops improving, wait N epochs before quitting')
    parser.add_argument('--beam_width', type=int, default=2,
                        help='number of particles to preserve at each timestep')
    parser.add_argument('--featext_layer_index', type=int, default=-1,
                        help='index of layer in vgg19 for feature extraction. Pass -1 to use all layers...')
    parser.add_argument('--sampling_prior', type=str, default='gaussian',
                        help='gaussian|angle')
    parser.add_argument('--angle_std', type=int, default=60,
                        help='std for angles when sampling_prior == angle')
    parser.add_argument('--avg_pool', action='store_true',
                        help='if true, replace MaxPool2D with AvgPool2D in VGG19')
    parser.add_argument('--segment_cost', type=float, default=0.0,
                        help='cost for drawing a single segment')
    args = parser.parse_args()

    assert args.beam_width <= args.n_samples
    assert args.featext_layer_index >= -1 and args.featext_layer_index < 45
    assert args.sampling_prior in ALLOWABLE_SAMPLING_PRIORS

    # pretrained on imagenet
    vgg19 = models.vgg19(pretrained=True)

    if args.avg_pool:  # replace max pool with avg pool
        vgg19 = vgg_convert_to_avg_pool(vgg19)

    if args.featext_layer_index == -1:  # use all layers
        vgg19 = VGG19Split(vgg19)
    else:  # cut off part of the net to generate features
        vgg19, _ = build_vgg19_feature_extractor(
            vgg19, chop_index=args.featext_layer_index)

    print('loaded vgg19')

    # needed for imagenet
    preprocessing = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # load the natural image we want to sketch
    img = Image.open(args.image_path)
    natural = Variable(preprocessing(img).unsqueeze(0))
    print('loaded natural image')

    # load the distractors that we use to normalize our metric
    distractor_paths = [os.path.join(args.distract_dir, i)
                        for i in os.listdir(args.distract_dir)]
    distractors = load_images_to_torch(distractor_paths, preprocessing)
    print('loaded distraction images\n')

    # use 1 layer if user specified else use all (8)
    n_features = 8 if featext_layer_index == -1 else 1
    n_distractors = len(distractor_paths)

    x_init, y_init = 112, 112  # start in center
    # store the best & average beam loss as we traverse
    best_loss_per_iter, average_loss_per_iter = [], []

    # variables for early-stopping by loss
    best_loss = -np.inf
    best_loss_iter = 0
    best_loss_beam = 0
    patience = args.patience

    # each queue holds future paths to try
    # for the first iteration, use beam_width repetitions of the init point
    # - a little inefficient but simple!
    x_beam_queue, y_beam_queue = [x_init]*args.beam_width, [y_init]*args.beam_width
    # store full path for top particles
    x_beam_paths = [[x_init] for b in range(args.beam_width)]
    y_beam_paths = [[y_init] for b in range(args.beam_width)]
    action_beam_paths = [[] for b in range(args.beam_width)]
    segment_cost = 0.0  # tracks cost for adding a segment

    for iter in range(args.n_iters):
        assert len(x_beam_queue) == len(y_beam_queue)
        print('ITER (%d/%d) - BEAM QUEUE LENGTH: %d' %
              (iter + 1, args.n_iters, args.beam_width))

        beam_losses = []  # stores losses across all beams
        x_beam_samples, y_beam_samples, action_beam_samples = [], [], []

        # beam search across a fixed width
        for b in range(args.beam_width):
            # sample new coordinates in 2 ways
            # 1. gaussian around the previous coordinate
            # 2. gaussian away from the previous coordinate with a given angle
            if args.sampling_prior == 'gaussian' or iter == 0:  # 1st iter has to be gaussian
                coord_samples = sample_endpoint_gaussian2d(x_beam_queue[b], y_beam_queue[b],
                                                           std=args.std, size=args.n_samples,
                                                           min_x=0, max_x=224, min_y=0, max_y=256)
            elif args.sampling_prior == 'angle':
                coord_samples = sample_endpoint_angle(x_beam_queue[b], y_beam_queue[b],
                                                      # estimate local context by the last 3 drawn points!
                                                      np.mean(x_beam_paths[b][-3:]),
                                                      np.mean(y_beam_paths[b][-3:]),
                                                      std=args.std, angle_std=args.angle_std,
                                                      size=args.n_samples, min_x=0, max_x=256,
                                                      min_y=0, max_y=224)

            x_samples, y_samples = coord_samples[:, 0].tolist(), coord_samples[:, 1].tolist()
            print('- generated %d samples' % args.n_samples)

            # sample an action e.g. draw vs move pen
            action_sample = gen_action()  # TODO: make probabilistic
            print('- sampled \'%s\' action' % action_sample)

            # the cost of drawing a new line increases over time
            if action_sample == 'draw':
                segment_cost += args.segment_cost

            # for each sample, render the image as a matrix
            sketches = []
            for i in range(args.n_samples):
                action_path = action_beam_paths[b] + [action_sample]
                renderer = RenderNet(x_beam_paths[b][-1], y_beam_paths[b][-1],
                                     x_samples[i], y_samples[i], imsize=224)
                sketch = renderer.forward()
                # manually apply preprocessing
                sketch[0] = (sketch[0] - 0.485) / 0.229
                sketch[1] = (sketch[1] - 0.456) / 0.224
                sketch[2] = (sketch[2] - 0.406) / 0.225
                sketches.append(sketch)

            # calculate embeddings for each image
            vgg19.eval()
            natural_emb = pytorch_batch_exec(vgg19, natural, args.batch_size,
                                             out_dim=n_features)
            distractor_embs = pytorch_batch_exec(vgg19, distractors, args.batch_size,
                                                 out_dim=n_features)
            sketch_embs = pytorch_batch_exec(vgg19, sketches, args.batch_size,
                                             out_dim=n_features)

            # compute losses
            losses = torch.zeros(args.n_samples)
            loss_fun = SketchLoss(n_features, n_distractors, segment_cost=segment_cost)
            for i in range(args.n_samples):
                loss = loss_fun(natural_emb, sketch_emb[i], distractor_embs)
                losses[i] = loss
            print('- computed losses')

            beam_losses += losses
            x_beam_samples += x_samples
            y_beam_samples += y_samples
            action_beam_samples.append(action_sample)
            print('Finished beam particle %d\n' % (b + 1))

        # make them all numpy arrays for easy indexing
        beam_losses = np.array(beam_losses)
        x_beam_samples = np.array(x_beam_samples)
        y_beam_samples = np.array(y_beam_samples)
        action_beam_samples = np.array(action_beam_samples)

        # keep the top N particles
        top_indexes = np.argsort(beam_losses)[::-1][:args.beam_width]

        # these will hold updated beam paths
        _x_beam_paths, _y_beam_paths, _action_beam_paths = [], [], []
        for b in range(args.beam_width):  # update our beam paths
            beam_parent = int(np.floor(top_indexes[b] / args.n_samples))  # which beam it originated from
            _x_beam_paths.append(x_beam_paths[beam_parent] + [x_beam_samples[top_indexes[b]]])
            _y_beam_paths.append(y_beam_paths[beam_parent] + [y_beam_samples[top_indexes[b]]])
            _action_beam_paths.append(action_beam_paths[beam_parent] + [action_beam_samples[b]])

        # overwrite old paths with new ones
        x_beam_paths = _x_beam_paths
        y_beam_paths = _y_beam_paths
        action_beam_paths = _action_beam_paths

        # update beam queue with top beam_width samples across beams
        x_beam_queue = [x_beam_samples[top_indexes[b]] for b in range(args.beam_width)]
        y_beam_queue = [y_beam_samples[top_indexes[b]] for b in range(args.beam_width)]

        # save loss statistics
        top_beam_loss = beam_losses[top_indexes[0]]
        average_beam_loss = np.mean(beam_losses[top_indexes])
        best_loss_per_iter.append(top_beam_loss)
        average_loss_per_iter.append(average_beam_loss)

        # update patience constants (and possibly break)
        if top_beam_loss >= best_loss:
            best_loss = top_beam_loss
            best_loss_beam = int(np.floor(top_indexes[0] / args.n_samples))
            best_loss_iter = iter
            patience = args.patience
            print('- new best loss %f. patience reset to %d' % (best_loss, patience))
        else:
            patience -= 1
            print('- no improvement %f. patience lowered to %d' % (best_loss, patience))

        if patience <= 0:  # this is equivalent to "stop" action
            print('- out of patience. quitting...')
            break

        print('')  # new line

    print('------------------------------')
    # delete generated files
    generated_paths = svg_paths + png_paths
    for path in generated_paths:
        os.remove(path)
    print('deleted %i generated files' % len(generated_paths))

    # save output to savedir
    save_path = os.path.join(args.sketch_dir, 'sketch.svg')
    # regenerate the canvas with strokes up to the best iter (before losing patience)
    svg_path = gen_canvas(x_beam_paths[best_loss_beam][:best_loss_iter+1],
                          y_beam_paths[best_loss_beam][:best_loss_iter+1],
                          action_beam_paths[best_loss_beam][:best_loss_iter],
                          outpath=save_path)
    png_path = svg2png(svg_path)
    os.remove(svg_path)
    print('saved sketch to \'%s\'' % png_path)

    # save the internal components:
    output_path = os.path.join(args.sketch_dir, 'sketch_outputs.npy')
    np.save(
        output_path,
        {
            'x': x_beam_paths[best_loss_beam][:best_loss_iter+1],
            'y': y_beam_paths[best_loss_beam][:best_loss_iter+1],
            'action': action_beam_paths[best_loss_beam][:best_loss_iter],
            'best_loss': best_loss,
            'patience': patience,
        },
    )
    print('saved output variables to \'%s\'' % output_path)

    # save plot to pngs
    plot_path = os.path.join(args.sketch_dir, 'distance_over_time.png')
    plt.figure()
    plt.plot(best_loss_per_iter, label='best')
    plt.plot(average_loss_per_iter, label='average')
    plt.xlabel('epoch')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path)
    print('saved losses to \'%s\'' % plot_path)
