from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import torch
import torch.nn as nn
import numpy as np
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
    return np.random.choice(action_state, 1, p=action_proba)[0]


def gen_canvas(x_list, y_list, action_list,
               outpath='./tmp.svg', size=256):
    """Draw a sequence of coordinates onto a canvas.

    :param x_list: list of floats - x coordinates
    :param y_list: list of floats - y coordinates
    :param action_list: list of strings ('draw', 'move', 'stop')
    :param outpath: string - where to save canvas
    :param size: size of the canvas
    """
    assert len(x_list) == len(y_list)
    assert len(x_list) == len(action_list) + 1
    assert set(action_list) == set('draw', 'move', 'stop')

    # initialize canvas
    size ='{}px'.format(size)
    dwg = svgwrite.Drawing(filename='../sketch/1.svg', size=(size, size))
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
    return imgs


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
    cov = np.eye(2) * std
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
    from PIL import Image

    parser = argparse.ArgumentParser(description="generate sketches")
    parser.add_argument('--imagepath', type=str, help='path to image file')
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
    data = Variable(preprocessing(img).unsqueeze(0))

    # cut off part of the net to generate features
    vgg_ext = chop_vgg19(vgg19)

    def extract_features(data):
        features = vgg_ext(data).data[0].numpy()
        return features / np.linalg.norm(features)

    def extract_scores(data):
        return vgg19(data).data[0].numpy()
