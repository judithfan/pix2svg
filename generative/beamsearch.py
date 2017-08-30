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
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

from linerender import LineRenderNet
from embeddings import VGG19Embeddings, ResNet152Embeddings

ALLOWABLE_POOLS = ['max', 'average']
ALLOWABLE_DISTANCE_FNS = ['cosine', 'euclidean', 'squared_euclidean',
                          'normalized_squared_euclidean', 'manhattan',
                          'chessboard', 'bray_curtis', 'canberra',
                          'correlation', 'binary']
ALLOWABLE_EMBEDDING_NETS = ['vgg19', 'resnet152']


class BaseBeamSearch(object):
    """Basic beam search algorithm. Lacks a loss function and has an
    empty preprocessing script. This performs successive prediction
    for endpoints (x1...n, y1...n) given a starting point (x0, y0).

    Contains a function train() that runs 1 iteration.

    :param x0: starting x coord (int)
    :param y0: starting y coord (int)
    :param imsize: square image size (int)
    :param beam_width: number of branches (int)
    :param n_samples: number of samples to draw (int)
    :param n_iters: number of iterations to allocate variables for (int)
    :param patience: number of steps to run before quitting w/o lower loss (int)
    :param stdev: standard deviation for the sampling Gaussian (float)
    :param fuzz: hyperparameter for rendering (float)
    :param fine_tune: if True, for each sampled endpoint, follow local
                      gradients to get the best segment possible (bool)
    """

    def __init__(self, x0, y0, imsize, beam_width=2, n_samples=100,
                 n_iters=10, patience=5, stdev=2, fuzz=1.0, fine_tune=False,
                 fine_tune_params={}, use_cuda=False):
        assert stdev > 0
        assert fuzz > 0
        assert patience >= 0
        assert imsize > 0
        assert n_samples > 0
        assert beam_width > 0
        assert x0 >= 0 and x0 < imsize
        assert y0 >= 0 and y0 < imsize

        self.x_beam_queue = np.ones(beam_width) * x0
        self.y_beam_queue = np.ones(beam_width) * y0

        self.x_beam_paths = np.zeros((beam_width, n_iters + 1))
        self.y_beam_paths = np.zeros((beam_width, n_iters + 1))
        self.action_beam_paths = [[] for b in range(beam_width)]
        # init path with starting point
        self.x_beam_paths[:, 0] = x0
        self.y_beam_paths[:, 0] = y0

        # save parameters
        self.beam_width = beam_width
        self.n_samples = n_samples
        self.patience = patience
        self.cur_patience = patience
        self.best_loss = np.inf
        self.n_iters = n_iters
        self.stdev = stdev
        self.imsize = imsize
        self.fuzz = fuzz
        self.fine_tune = fine_tune
        self.fine_tune_params = fine_tune_params
        self.use_cuda = use_cuda
        self.top_index = 0

        self.beam_sketches = Variable(torch.zeros((beam_width, 1, imsize, imsize)))
        if use_cuda:
            self.beam_sketches = self.beam_sketches.cuda()

    def gen_paths(self):
        b = int(np.floor(self.top_index / self.n_samples))
        return (self.x_beam_paths[b], self.y_beam_paths[b])

    def sketch_loss(self, input_item, pred_items, distractor_items=None):
        raise NotImplementedError

    def preprocess_sketches(self, sketches):
        return sketches

    def update_patience(self, loss):
        if loss >= self.best_loss:
            self.best_loss = loss
            self.cur_patience = patience
        else:
            self.cur_patience -= 1

    def tune(self, epoch, renderer, optimizer, input_item,
             distractor_items=None, verbose=False):
        renderer.train()
        optimizer.zero_grad()
        sketch = renderer()
        losses = self.sketch_loss(input_item, sketch, distractor_items=distractor_items,
                                  use_cuda=self.use_cuda)
        loss = torch.sum(losses)
        loss.backward()
        optimizer.step()
        if verbose:
            params = list(renderer.parameters())
            print('Fine Tuning Epoch: {} \tLoss: {:.6f} \tParams: ({}, {})'.format(
                  epoch, loss.data[0], params[0].data.numpy()[0], params[1].data.numpy()[0]))

    def train(self, epoch, input_item, distractor_items=None):
        for b in range(self.beam_width):
            samples = sample_endpoint_gaussian2d(self.x_beam_queue[b], self.y_beam_queue[b],
                                                 std=self.stdev, size=self.n_samples,
                                                 min_x=0, max_x=self.imsize,
                                                 min_y=0, max_y=self.imsize)
            x_samples, y_samples = samples[:, 0], samples[:, 1]
            action_sample = sample_action()

            sketches = Variable(torch.zeros((self.n_samples, 1, self.imsize, self.imsize)),
                                volatile=True)  # no training in vgg
            if self.use_cuda:
                sketches = sketches.cuda()
            
            for i in range(self.n_samples):
                action_path = self.action_beam_paths[b] + [action_sample]
                renderer = LineRenderNet(self.x_beam_paths[b][epoch], self.y_beam_paths[b][epoch],
                                         x_samples[i], y_samples[i], imsize=self.imsize, 
                                         fuzz=self.fuzz, use_cuda=self.use_cuda)
                if self.use_cuda:
                    renderer.cuda()

                if self.fine_tune:
                    print('Fine Tuning Sample [{}/{}]'.format(i + 1, self.n_samples))
                    tune_lr = self.fine_tune_params.get('lr', 1e-2)
                    tune_momentum = self.fine_tune_params.get('momentum', 0.5)
                    tune_iters = self.fine_tune_params.get('n_iters', 100)
                    tune_log_interval = self.fine_tune_params.get('log_interval', 50)
                    tune_fuzz = self.fine_tune_params.get('fuzz', 1.0)

                    model = LineRenderNet(self.x_beam_paths[b][epoch], self.y_beam_paths[b][epoch],
                                          x_samples[i], y_samples[i], imsize=self.imsize, 
                                          fuzz=tune_fuzz, use_cuda=self.use_cuda)
                    if self.use_cuda:
                        model.cuda()

                    optimizer = optim.SGD(model.parameters(), lr=tune_lr, momentum=tune_momentum)
                    # wiggle the segment using the gradient to get a better fit
                    for iter in range(tune_iters):
                        self.tune(iter, model, optimizer, input_item, distractor_items,
                                  verbose=iter % tune_log_interval == 0)
                    print('')

                    # update renderer using model's parameters
                    renderer = model

                sketch = renderer()
                sketch = torch.add(sketch, self.beam_sketches[b])
                sketch_min = torch.min(sketch).expand_as(sketch)
                sketch_max = torch.max(sketch).expand_as(sketch)
                sketch = (sketch - sketch_min) / (sketch_max - sketch_min)
                sketches[i] = sketch

            sketches_raw = sketches.clone()
            sketches = self.preprocess_sketches(sketches)
            losses = self.sketch_loss(input_item, sketches, distractor_items, 
                                      use_cuda=self.use_cuda)

            if b == 0:
                beam_losses = losses.cpu().data.numpy()[0]
                x_beam_samples = x_samples
                y_beam_samples = y_samples
                action_beam_samples = np.array([action_sample])
                all_sketches = sketches_raw.clone()
            else:
                beam_losses = np.concatenate((beam_losses, losses.data.numpy()[0]))
                x_beam_samples = np.concatenate((x_beam_samples, x_samples))
                y_beam_samples = np.concatenate((y_beam_samples, y_samples))
                action_beam_samples = np.append(action_beam_samples, action_sample)
                all_sketches = torch.cat((all_sketches, sketches_raw.clone()), dim=0)

        top_indexes = np.argsort(beam_losses)[:self.beam_width]
        _beam_sketches = Variable(torch.zeros((self.beam_width, 1, self.imsize, self.imsize)))

        for b in range(self.beam_width):
            # not each beam will update...
            beam_parent = int(np.floor(top_indexes[b] / self.n_samples))

            x_beam_path = self.x_beam_paths[beam_parent, :]
            y_beam_path = self.y_beam_paths[beam_parent, :]
            x_beam_path[epoch + 1] = x_beam_samples[top_indexes[b]]
            y_beam_path[epoch + 1] = y_beam_samples[top_indexes[b]]
            self.x_beam_paths[b, :] = x_beam_path
            self.y_beam_paths[b, :] = y_beam_path
            self.action_beam_paths[b].append(action_beam_samples[b])
            _beam_sketches[b] = all_sketches[top_indexes[b]]

        self.beam_sketches = _beam_sketches  # replace old with new
        self.x_beam_queue = np.array([x_beam_samples[top_indexes[b]]  # recreate queue
                                      for b in range(self.beam_width)])
        self.y_beam_queue = np.array([y_beam_samples[top_indexes[b]]
                                      for b in range(self.beam_width)])
        self.update_patience(beam_losses[top_indexes[0]])

        print('Train Epoch: {} \tLoss: {:.6f} \tParams: ({}, {}) \tPatience: {}'.format(
              epoch, beam_losses[top_indexes[0]], x_beam_samples[top_indexes[0]],
              y_beam_samples[top_indexes[0]], self.cur_patience))

        if self.cur_patience <= 0:
            print('Out of patience. Exited.')
            return all_sketches[top_indexes[0]]

        self.top_index = top_indexes[0]  # save top index
        return all_sketches[top_indexes[0]]


class PixelBeamSearch(BaseBeamSearch):
    """Beam search with a pixel-wise loss"""

    def sketch_loss(self, input_item, pred_items, distractor_items=None, use_cuda=False):
        return pixel_sketch_loss(input_item, pred_items, distractor_images=distractor_items,
                                 use_cuda=use_cuda)


class SemanticBeamSearch(BaseBeamSearch):
    """Beam search with a semantic VGG19-layer loss

    :param embedding_layer: layer to use for embeddings; pass -1 for all layers (int)
    :param distance_fn: type of distance metric (cosine|l1|l2)
    """

    def __init__(self, x0, y0, imsize, beam_width=2, n_samples=100,
                 n_iters=10, stdev=2, fuzz=1.0, distance_fn='cosine',
                 embedding_net='vgg19', embedding_layer=-1, use_cuda=False):
        super(SemanticBeamSearch, self).__init__(x0, y0, imsize, beam_width=beam_width,
                                                 n_samples=n_samples, n_iters=n_iters,
                                                 stdev=stdev, fuzz=1.0, use_cuda=use_cuda)
        assert embedding_net in ALLOWABLE_EMBEDDING_NETS

        if embedding_net == 'vgg19':
            assert embedding_layer >= -1 and embedding_layer < 8
            self.embedding_net = load_vgg19(layer_index=embedding_layer,
                                            use_cuda=use_cuda)
        elif embedding_net == 'resnet152':
            assert embedding_layer >= -1 and embedding_layer < 7
            self.embedding_net = load_resnet152(layer_index=embedding_layer,
                                                use_cuda=use_cuda)
        if use_cuda:
            self.embedding_net.cuda()

    def preprocess_sketches(self, sketches):
        sketches = torch.cat((sketches, sketches, sketches), dim=1)
        sketches[:, 0] = (sketches[:, 0] - 0.485) / 0.229
        sketches[:, 1] = (sketches[:, 1] - 0.456) / 0.224
        sketches[:, 2] = (sketches[:, 2] - 0.406) / 0.225
        return self.embedding_net(sketches)  # return embeddings

    def sketch_loss(self, input_item, pred_items, distractor_items=None, use_cuda=False):
        return semantic_sketch_loss(input_item, pred_items, distractor_embs=distractor_items,
                                    distance_fn='cosine', use_cuda=use_cuda)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    r"""Returns cosine similarity between x1 and x2, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.cosine_similarity(input1, input2)
        >>> print(output)
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def gen_distance(a, b, metric='cosine'):
    """Implementation of difference distance metrics ripped from Wolfram:
    http://reference.wolfram.com/language/guide/DistanceAndSimilarityMeasures.html
    """
    if metric == 'cosine':
        return cosine_similarity(a, b, dim=1)
    elif metric == 'euclidean':
        return torch.norm(a - b, p=2)
    elif metric == 'squared_euclidean':
        return torch.pow(torch.norm(a - b, p=2))
    elif metric == 'normalized_squared_euclidean':
        c = a - torch.mean(a, dim=1)
        d = b - torch.mean(b, dim=1)
        n = torch.pow(torch.norm(c, p=2)) + torch.pow(torch.norm(d, p=2))
        return 0.5 * torch.pow(torch.norm(c - d, p=2)) / n
    elif metric == 'manhattan':
        return F.pairwise_distance(a, b, p=1)
    elif metric == 'chessboard':
        return torch.max(torch.abs(a - b))
    elif metric == 'bray_curtis':
        return torch.sum(torch.abs(a - b)) / torch.sum(torch.abs(a + b))
    elif metric == 'canberra':
        return torch.sum(torch.abs(a - b) / (torch.abs(a) + torch.abs(b)))
    elif metric == 'correlation':
        c = a - torch.mean(a, dim=1)
        d = b - torch.mean(b, dim=1)
        return F.cosine_similarity(c, d)
    elif metric == 'binary':
        return torch.sum(a != b)


def pixel_sketch_loss(natural_image, sketch_images, distractor_images=None,
                      segment_cost=0.0, use_cuda=False):
    """Calculate L2 distance between natural image and sketch
    image + normalization using distractor images.

    :param natural_image: PyTorch Tensor 1xCxHxW
    :param sketch_images: PyTorch Tensor SxCxHxW
    :param distractor_images: PyTorch Tensor DxCxHxW (default None)
    :param segment_cost: cost of adding this segment (default 0.0)
    :param use_cuda: create variables with cuda 
    :return loss: vector of size sketch_images.size(0)
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


def semantic_sketch_loss(natural_emb, sketch_embs, distractor_embs=None,
                         distance_fn='cosine', segment_cost=0.0, use_cuda=False):
    """Calculate distance between natural image and sketch image
    where the distance is normalized by distractor images.

    :param natural_emb: tuple of PyTorch Variable
    :param sketch_embs: tuple of PyTorch Variables
    :param distractor_embs: tuple of PyTorch Variables (default None)
    :param distance_fn: string defining the type of distance function to use (default cosine)
    :param segment_cost: cost of adding this segment (default 0)
    :param use_cuda: create variables with cuda
    :return loss: vector size of sketch_embs.size(0)
    """
    assert distance_fn in ALLOWABLE_DISTANCE_FNS
    n_sketches = sketch_embs[0].size()[0]
    n_features = len(natural_emb)

    loss = Variable(torch.zeros(n_sketches), requires_grad=True)
    if use_cuda:
        loss = loss.cuda()

    for f in range(n_features):
        costs = gen_distance(natural_emb[f].expand_as(sketch_embs[f]), sketch_embs[f], 
                             metric=distance_fn)
        loss = torch.add(loss, costs)

    if distractor_embs is not None:
        n_distractors = distractor_embs[0].size()[0]
        distraction_dists = Variable(torch.zeros((n_distractors, n_sketches)))
        if use_cuda:
            distraction_dists = distraction_dists.cuda()
        for j in range(n_distractors):
            for f in range(n_features):
                distraction_emb = torch.unsqueeze(distractor_embs[f][j], dim=0)
                costs = gen_distance(distraction_emb.expand_as(sketch_embs[f]), sketch_embs[f], 
                                     metric=distance_fn)
                distraction_dists[j] = torch.add(distraction_dists[j], costs)

        all_dists = torch.cat((torch.unsqueeze(loss, dim=0), distraction_dists))
        norm = torch.norm(all_dists, p=2, dim=0)
        loss = loss / norm

    return loss + segment_cost


def sample_action():
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
    return samples


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
    return samples


def load_resnet152(layer_index=-1, use_cuda=False):
    resnet152 = models.resnet152(pretrained=True)
    resnet152 = ResNet152Embeddings(resnet152, layer_index)
    resnet152.eval()  # freeze dropout

    # freeze each parameter
    for p in resnet152.parameters():
        p.requires_grad = False

    return resnet152


def load_vgg19(layer_index=-1, use_cuda=False):
    vgg19 = models.vgg19(pretrained=True)
    vgg19 = VGG19Embeddings(vgg19, layer_index)
    vgg19.eval()  # freeze dropout

    # freeze each parameter
    for p in vgg19.parameters():
        p.requires_grad = False

    return vgg19
