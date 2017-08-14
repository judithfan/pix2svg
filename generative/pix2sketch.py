from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import math
import svgwrite
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


def gen_interpretable_label(label_index):
    with open('class_index.json') as fp:
        class_idx = json.load(fp)
        idx2label = [class_idx[str(k)][1]
                     for k in range(len(class_idx))]
    return idx2label[label_index]


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


def gen_action():
    """Sample an action from an action space of 'draw' or 'move'
    For now, we simply always return 'draw'.
    """
    return 'draw'


def gen_canvas(x_list, y_list, action_list, outpath='./tmp.svg', size=256):
    """Draw a sequence of coordinates onto a canvas.

    :param x_list: list of floats - x coordinates
    :param y_list: list of floats - y coordinates
    :param action_list: list of strings ('draw', 'move')
    :param outpath: string - where to save canvas
    :param size: size of the canvas
    """
    assert len(x_list) == len(y_list)
    assert len(x_list) == len(action_list) + 1
    assert set(action_list) <  set(['draw', 'move'])

    # initialize canvas
    size ='{}px'.format(size)
    dwg = svgwrite.Drawing(filename=outpath, size=(size, size))
    dwg.add(dwg.rect(insert=(0, 0),
                 size=(size, size),
                 fill='white'))

    # start with first point and draw/move a segment
    num_coords = len(x_list)
    for i in range(num_coords - 1):
        x_s, y_s = x_list[i], y_list[i]
        x_e, y_e = x_list[i + 1], y_list[i + 1]
        if action_list[i] == 'draw':
            path = "M {x_s},{y_s} L {x_e},{y_e} ".format(x_s=x_s, y_s=y_s,
                                                         x_e=x_e, y_e=y_e)
        elif action_list[i] == 'move':
            path = "M {x_e},{y_e} ".format(x_e=x_e, y_e=y_e)
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


def compute_losses(natural_image, sketch_images, distraction_images, vgg_features,
                   batch_size=100):
    """Compute a loss to describe how semantically far apart
    the natural image and the sketch image are. If label is not
    None, compute an additional penalty for getting the label wrong.

    loss = distance(natural_image, sketch_image, distraction_images)

    Compute the losses for each (natural_image, sketch_image) pair and
    normalize each loss by each (distraction_image, sketch_image) pair

    :param natural_image: PyTorch Variable (1 x C x H x W)
    :param sketch_images: PyTorch Variable (N x C x H x W)
    :param distraction_images: PyTorch Variable (M x C x H x W)
    :param vgg_features: VGG19 PyTorch instance that produces features
    :param batch_size: run X images at a time through deep net
    """

    def extract_features(data):
        """:param data: 4d array of NxCxHxW"""
        vgg_features.eval()
        features = vgg_features(data).data.numpy()
        return features / np.linalg.norm(features, axis=1)[:, None]

    def extract_scores(data):
        """:param data: 4d array of NxCxHxW"""
        vgg_scores.eval()
        scores = np.exp(vgg_scores(data).data.numpy())
        return scores / np.linalg.norm(scores, axis=1)[:, None]

    natural_features = extract_features(natural_image)
    # compute features for all distraction images (not batching since this is a small number)
    num_distractions = distraction_images.size()[0]
    distraction_features_arr = extract_features(distraction_images)

    # compute features for all sketches in batches (if batch is too
    # large then we will have memory issues)
    num_sketches = sketch_images.size()[0]  # number of images total
    num_reads = int(math.floor(num_sketches / batch_size))  # number of passes needed
    num_processed = 0  # track the number of batches processed
    sketch_features_arr = []

    for i in range(num_reads):
        sketch_images_batch = sketch_images[
            num_processed:num_processed+batch_size,
        ]
        sketch_features_batch = extract_features(sketch_images_batch)
        sketch_features_arr.append(sketch_features_batch)
        num_processed += batch_size

    # process remaining images
    if num_sketches - num_processed > 0:
        sketch_images_batch = sketch_images[
            num_processed:num_sketches,
        ]
        sketch_features_batch = extract_features(sketch_images_batch)
        sketch_features_arr.append(sketch_features_batch)

    # stack all of them together
    sketch_features_arr = np.vstack(sketch_features_arr)

    # for each sketch, compute loss with natural image
    losses = [0]*num_sketches
    for i in range(num_sketches):
        natural_dist = 1 - cosine(natural_features, sketch_features_arr[i])
        # these distractions serve as context such that we can ground the natural_dist metric
        # into some interpretable space.
        distraction_dists = []
        for j in range(num_distractions):
            distraction_dists.append(1 - cosine(distraction_features_arr[j], sketch_features_arr[i]))
        # normalize natural_dist by distraction_dists
        all_dists = np.array([natural_dist] + distraction_dists)
        losses[i] = natural_dist / np.linalg.norm(all_dists)

    return losses


if __name__ == '__main__':
    import json
    import argparse

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')

    parser = argparse.ArgumentParser(description="generate sketches")
    parser.add_argument('--image_path', type=str, help='path to image file')
    parser.add_argument('--distract_dir', type=str, help='directory to distractor image files')
    parser.add_argument('--sketch_dir', type=str, help='directory to store sketches')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='number of samples per iteration')
    parser.add_argument('--n_iters', type=int, default=20,
                        help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='size when running forward pass (memory issues)')
    parser.add_argument('--std', type=float, default=15.0,
                        help='std for Gaussian when sampling')
    parser.add_argument('--patience', type=int, default=5,
                        help='once the informativity measure stops improving, wait N epochs before quitting')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='number of particles to preserve at each timestep')
    args = parser.parse_args()

    # pretrained on imagenet
    vgg19 = models.vgg19(pretrained=True)
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

    # load the distractions that we use to normalize our metric
    distraction_paths = [os.path.join(args.distract_dir, i)
                         for i in os.listdir(args.distract_dir)]
    distractions = load_images_to_torch(distraction_paths, preprocessing)
    print('loaded distraction images\n')

    # cut off part of the net to generate features
    vgg_ext = build_vgg19_feature_extractor(vgg19)

    x_init, y_init = 128, 128  # start in center
    # store the best & average beam loss as we traverse
    best_loss_per_iter, average_loss_per_iter = [], []

    # variables for early-stopping by loss
    best_loss = 0.0
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

    for iter in range(args.n_iters):
        assert len(x_beam_queue) == len(y_beam_queue)
        print('ITER (%d/%d) - BEAM QUEUE LENGTH: %d' %
              (iter + 1, args.n_iters, args.beam_width))

        beam_losses = []  # stores losses across all beams
        x_beam_samples, y_beam_samples, action_beam_samples = [], [], []

        for b in range(args.beam_width):
            # load particle coordinates
            coord_samples = sample_endpoints(x_beam_queue[b], y_beam_queue[b],
                                             std=args.std, size=args.n_samples,
                                             min_x=0, max_x=256, min_y=0, max_y=256)
            x_samples, y_samples = coord_samples[:, 0].tolist(), coord_samples[:, 1].tolist()
            print('- generated %d samples' % args.n_samples)

            action_sample = gen_action()  # TODO: make probabilistic
            print('- sampled \'%s\' action' % action_sample)

            svg_paths = []  # try out each sample for the current particle
            for i in range(args.n_samples):
                x_path = x_beam_paths[b] + [x_samples[i]]
                y_path = y_beam_paths[b] + [y_samples[i]]
                action_path = action_beam_paths[b] + [action_sample]
                # create svg drawing
                svg_path = gen_canvas(x_path, y_path, action_path,
                                      outpath=os.path.join(args.sketch_dir,
                                                           'iter_{}_beam_{}.svg'.format(i, b)))
                svg_paths.append(svg_path)
            print('- generated %i svg canvases' % args.n_samples)

            # convert svg to pngs
            png_paths = [svg2png(svg_paths[i]) for i in range(args.n_samples)]
            print('- converted to %i png canvases' % args.n_samples)

            sketches = load_images_to_torch(png_paths, preprocessing)
            losses = compute_losses(natural, sketches, distractions, vgg_ext,
                                    batch_size=args.batch_size)

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
            _y_beam_paths.append(x_beam_paths[beam_parent] + [y_beam_samples[top_indexes[b]]])
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
            best_loss_beam = top_indexes[0] % args.n_samples
            best_loss_iter = iter
            patience = args.patience
            print('- new best loss %f. patience reset to %d' % (best_loss, patience))
        else:
            patience -= 1
            print('- no improvement. patience lowered to %d' % patience)

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
