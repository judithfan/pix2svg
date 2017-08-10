from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import cosine


def build_vgg19_feature_extractor(vgg):
    """Take features from the activations of the hidden layer
    immediately before the VGG's object classifier (4096 size).
    The last linear layer and last dropout layer are removed,
    preserving the ReLU. The activations are L2 normalized.

    :param vgg: trained vgg19 model
    :return: PyTorch Sequential model
    """
    vgg_copy = copy.deepcopy(vgg)
    classifier = nn.Sequential(*list(vgg.classifier.children())[:-2])
    vgg_copy.classifier = classifier
    return vgg_copy


def gen_action(iter, min_iter=10, max_iter=20, min_stop_proba=0.0,
               max_stop_proba=0.50, draw_move_ratio=0.75):
    """Sample an action from an action space of draw, move, stop.
    We reverse anneal the stop probability as the iter approaches
    the maximum iteration.

    :param iter: current iteration
    :param min_iter: when to start reverse annealing stop probabilities
    :param max_iter: when to reach the max stop probability
    :param min_stop_proba: beginning stop probability
    :param max_stop_proba: maximum stop probability
    :param draw_move_ratio: fixed ratio of draw to move probability
    """
    if iter < min_iter:
        stop_proba = min_stop_proba
        draw_proba = (1.0 - stop_proba) * draw_move_ratio
        move_proba = 1 - draw_proba - stop_proba
    else:
        stop_rate = (max_stop_proba - min_stop_proba) / (max_iter - min_iter)
        stop_proba = min_stop_proba + iter * stop_rate
        draw_proba = (1.0 - stop_proba) * draw_move_ratio
        move_proba = 1 - draw_proba - stop_proba
    action_proba = np.array([draw_proba, move_proba, stop_proba])
    action_state = np.array(['draw', 'move', 'stop'])
    return np.random.choice(action_state, 1, p=action_proba)


def gen_canvas(x_list, y_list, action_list, outpath='./tmp.svg', size=256):
    """Draw a sequence of coordinates onto a canvas.

    :param x_list: list of floats - x coordinates
    :param y_list: list of floats - y coordinates
    :param action_list: list of strings ('draw', 'move', 'stop')
    :param outpath: string - where to save canvas
    :param size: size of the canvas
    """
    assert len(x_list) == len(y_list)
    assert len(x_list) == len(action_list) + 1
    assert set(action_list) <  set(['draw', 'move', 'stop'])

    # initialize canvas
    size ='{}px'.format(size)
    dwg = svgwrite.Drawing(filename=outpath, size=(size, size))
    dwg.add(dwg.rect(insert=(0, 0),
                 size=(size, size),
                 fill='white'))

    # start with first point and draw/move/stop a segment
    x_s, y_s = x_list[0], y_list[0]
    x_list, y_list = x_list[1:], y_list[1:]
    num_coords = len(x_list)
    for i in range(num_coords):
        x_e, y_e = x_list[i], y_list[i]
        if action_list[i] == 'draw':
            path = "M {x_s},{y_s} L {x_e},{y_e} ".format(x_s=x_s, y_s=y_s,
                                                         x_e=x_e, y_e=y_e)
        elif action_list[i] == 'move':
            path = "M {x_e},{y_e} ".format(x_e=x_e, y_e=y_e)
        else:  # 'stop'
            continue
        # perform the action
        dwg.add(dwg.path(path).stroke("black", 3))

    # save canvas
    dwg.save()
    return outpath


def svg2png(svgpath, width=256, height=256):
    """Convert SVG files to PNGs"""
    svgname = os.path.basename(svgpath)
    directory = os.path.dirname(svgpath)
    pngpath = os.path.join(directory, svgname.replace('.svg', '.png'))
    cmd = 'rsvg -w {width} -h {height} "{svgpath}" -o "{pngpath}"'.format(
        width=width, height=height,
        svgpath=svgpath, pngpath=pngpath,
    )
    os.system(cmd)
    return pngpath


def load_canvases(paths):
    """Load a bunch saved canvas.svg into a Torch environment
    as a single Torch Variable.

    :param paths: list of str
    :return: Torch Variable
    """
    num_paths = len(paths)
    imgs = []
    for i in range(num_paths):
        img = Image.open(paths[i])
        imgs.append(preprocessing(img).unsqueeze(0))
    imgs = torch.cat(imgs)
    return Variable(imgs)


def sample_endpoints(x_s, y_s, std=10, size=1, min_x=0, max_x=256, min_y=0, max_y=256):
    """Sample a coordinate (x_e, y_e) from a 2D gaussian with set deviation.
    The idea is to bias coordinates closer to the start to make smaller strokes.

    :param x_s: starting x coordinate
    :param y_s: starting y coordinate
    :param std: default 10 - controls stroke size
    :param size:
    :param min_x: default 0
    :param max_x: default 256
    :param min_y: default 0
    :param max_y: default 256
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
    return samples


def compute_losses(natural_image, sketch_images, label=None, label_weight=1.0):
    """Compute a loss to describe how semantically far apart
    the natural image and the sketch image are. If label is not
    None, compute an additional penalty for getting the label wrong.

    loss = distance(natural_image, sketch_image) + label_weight*xentropy(proba, labels)

    Compute the losses for each (natural_image, sketch_image) pair.

    :param natural_image: PyTorch Variable (1 x C x H x W)
    :param sketch_image: PyTorch Variable (N x C x H x W)
    :param label: float - a number from 0 to # classes - 1
    :param label_weight: regularization parameter to weight xentropy
    """
    natural_features = extract_features(natural_image)
    # compute features for all sketches at once
    sketch_features_arr = extract_features(sketch_images)
    num_sketches = sketch_features_arr.shape[0]
    if label is not None:
        # if we are weighting the class, compute all scores at once
        sketch_class_probas_arr = extract_scores(sketch_images)

    # for each sketch, compute loss with natural image
    losses = np.zeros(num_sketches)
    for i in range(num_sketches):
        dist = 1 - cosine(natural_features, sketch_features_arr[i])
        if label is not None:
            dist += label_weight * F.nll_loss(
                Variable(torch.from_numpy(sketch_class_probas_arr[i][np.newaxis])),
                Variable(torch.from_numpy(np.array([label])))
            ).exp().data.numpy()[0]
        losses[i] = dist

    return losses


if __name__ == '__main__':
    import os
    import json
    import argparse

    parser = argparse.ArgumentParser(description="generate sketches")
    parser.add_argument('--imagepath', type=str, help='path to image file')
    parser.add_argument('--sketchdir', type=str, help='directory to store sketches')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='number of samples per iteration')
    parser.add_argument('--n_iters', type=int, default=20,
                        help='number of iterations')
    args = parser.parse_args()

    with open('class_index.json') as fp:
        class_idx = json.load(fp)
        idx2label = [class_idx[str(k)][1]
                     for k in range(len(class_idx))]

    # pretrained on imagenet
    vgg19 = models.vgg19(pretrained=True)

    # needed for imagenet
    preprocessing = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(args['imagepath'])
    natural = Variable(preprocessing(img).unsqueeze(0))

    # cut off part of the net to generate features
    vgg_ext = build_vgg19_feature_extractor(vgg19)

    def extract_features(data):
        """:param data: 4d array of NxCxHxW"""
        vgg_ext.eval()
        features = vgg_ext(data).data.numpy()
        return features / np.linalg.norm(features, axis=1)[:, None]

    def extract_scores(data):
        """:param data: 4d array of NxCxHxW"""
        vgg19.eval()
        scores = np.exp(vgg19(data).data.numpy())
        return scores / np.linalg.norm(scores, axis=1)[:, None]

    x_e, y_e = 128, 128  # start in center
    x_selected, y_selected, action_selected = [x_e], [y_e], []

    for iter in range(args['n_iters']):
        print('ITER (%i/%i):' % iter + 1, args['n_iters'])
        coord_samples = sample_endpoints(x_e, y_e, std=15, size=args['n_samples'],
                                         min_x=0, max_x=256, min_y=0, max_y=256)
        x_samples, y_samples = coord_samples[:, 0], coord_samples[:, 1]
        print('- generated %i samples' % args['n_samples'])
        # once we hit a stop, always stop
        if 'stop' in action_selected:
            action_samples = ['stop' for i in range(args['n_samples'])]
            print('- stop override')
        else:
            action_sample = gen_action(1)
            print('- sampled %s action' % action_sample)

        svg_paths = []
        for i in range(args['n_samples']):
            x_path = x_selected + [x_samples[i]]
            y_path = y_selected + [y_samples[i]]
            action_path = action_selected + [action_sample]
            svg_path = gen_canvas(x_path, y_path, action_path,
                                  outpath=os.path.join(args['sketchdir'], '{}.svg'.format(i)))
            svg_paths.append(svg_path)
        print('- generated %i svg canvases' % args['n_samples'])

        # do this in a separate for loop to avoid conflicts in render time
        png_paths = []  # convert svgs to pngs
        for i in range(args['n_samples']):
            png_path = svg2png(os.path.join(args['sketchdir'], '{}.svg'.format(i)))
            png_paths.append(png_path)
        print('- converted to %i png canvases' % args['n_samples'])

        sketches = load_canvases(png_paths)
        losses = compute_losses(natural, sketches)
        winning_index = np.argmax(losses)
        print('- calculated loss; best loss: %f' % max(losses))

        x_selected.append(x_samples[winning_index])
        y_selected.append(y_samples[winning_index])
        action_selected.append(action_sample)
        print('')  # new line

    print('------------------------------')
    # delete generated files
    generated_paths = svg_paths + png_paths
    for path in generated_paths:
        os.remove(path)
    print('deleted %i generated files' % len(generated_paths))

    # save output to savedir
    save_path = os.path.join(args['sketchdir'], 'sketch.svg')
    svg_path = gen_canvas(x_selected, y_selected, action_selected,
                          outpath=save_path)
    png_path = svg2png(svg_path)
    os.remove(svg_path)
    print('saved sketch to %s' % png_path)
