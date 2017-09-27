from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class SketchRenderNet(nn.Module):
    """Similar to LineRenderNet. This also renders an image
    as a CxHxW matrix but it is given a sequence of points
    (x0, x1, ..., xn), (y0, y1, .., yn) and all of them are
    parameters (no fixed points).

    :param x_list: path of x coordinates (x0, x1, ..., xn)
    :param y_list: path of y coordinates (y0, y1, ..., yn)
    :param pen_list: path of pen types (p0, p1, ..., pn)
    :param imsize: image size to generate
    :param n_params: make the last <n_params> parameters
    :param fuzz: hyperparameter to scale differences; fuzz > 1 would
                 localize around the line; fuzz < 1 would make things
                 more uniform.
    :param smoothness: the bigger it is, the closer it is to min() func.
    :param use_cuda: boolean to gen cuda variables
    :return template: imsize by imsize rendered sketch
    """
    def __init__(self, x_list, y_list, pen_list=None, imsize=224, 
                 n_params=-1, fuzz=1, smoothness=8, use_cuda=False):
        super(SketchRenderNet, self).__init__()
        assert len(x_list) == len(y_list)
        assert n_params == -1 or n_params > 1

        n_points = len(x_list)
        n_params = n_points if n_params == -1 else n_params
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        
        if pen_list is None:
            # if none is provided, draw everything.
            pen_list = [2 for i in xrange(n_points)]

        # we normalize the params to be between 0 and 1 so that its an 
        # easier optimization problem
        x_params = torch.Tensor(x_list[-n_params:]) / imsize
        y_params = torch.Tensor(y_list[-n_params:]) / imsize
        self.x_params = Parameter(x_params.type(dtype))
        self.y_params = Parameter(y_params.type(dtype))
        self.pen_params = pen_list[-n_params:]  # just a regular list

        # we will store distances from points to each segment 
        n_draws = sum(1 for i in pen_list if i == 2)
        draw_ix = 0  # stores index of draw
        template = torch.zeros(n_draws, imsize, imsize).type(dtype)
        
        # computed fixed parts if they exist
        if n_params < n_points:
            n_seeds = n_points - n_params
            x_fixed = torch.Tensor(x_list[:n_seeds]).type(dtype)
            y_fixed = torch.Tensor(y_list[:n_seeds]).type(dtype)
            pen_fixed = pen_list[:n_seeds]

            for i in range(1, n_seeds):
                if pen_fixed[i] == 2:
                    _template = draw_line(x_fixed[i - 1], y_fixed[i - 1], x_fixed[i], y_fixed[i],
                                          imsize=imsize, fuzz=fuzz, use_cuda=use_cuda)
                    template[draw_ix] = _template
                    draw_ix += 1

        self.template = template
        self.imsize = imsize
        self.fuzz = fuzz
        self.use_cuda = use_cuda
        self.n_params = n_params
        self.draw_ix = draw_ix
        self.smoothness = smoothness

    def forward(self): 
        draw_ix = self.draw_ix
        template = Variable(self.template)
        for i in range(1, self.n_params):
            if self.pen_params[i] == 2:
                # b/c our params are scaled to 0 --> 1, we need to resize them
                # back to 0 --> imsize
                _template = draw_line(self.x_params[i - 1] * self.imsize, 
                                      self.y_params[i - 1] * self.imsize,
                                      self.x_params[i] * self.imsize, 
                                      self.y_params[i] * self.imsize,
                                      imsize=self.imsize, fuzz=self.fuzz,
                                      use_cuda=self.use_cuda)
                template[draw_ix] = _template
                draw_ix += 1
        
        template = exponential_smooth_min(template, dim=0, k=self.smoothness)
        # add a dimension for batches and a dimension for channels
        template = torch.unsqueeze(template, dim=0)
        template = torch.unsqueeze(template, dim=0)
        
        return template


class LineRenderNet(nn.Module):
    """Renders an image as a CxHxW matrix given 2 points
    such that it is differentiable. The intensity of each
    pixel is the shortest distance from each pixel to the line.

    :param x0: fixed starting x coordinate
    :param y0: fixed starting y coordinate
    :param x1: initialization for ending x coordinate
    :param y1: initialization for ending y coordinate
    :param imsize: image size to generate
    :param fuzz: hyperparameter to scale differences; fuzz > 1 would
                 localize around the line; fuzz < 1 would make things
                 more uniform.
    :param use_cuda: make variables using cuda
    :return template: imsize by imsize rendered sketch
    """
    def __init__(self, x0, y0, x1, y1, imsize=224, fuzz=1, use_cuda=False):
        super(LineRenderNet, self).__init__()
        if use_cuda:
            self.x0 = Variable(torch.cuda.FloatTensor([x0]))
            self.y0 = Variable(torch.cuda.FloatTensor([y0]))
            self.x1 = Parameter(torch.cuda.FloatTensor([x1]))
            self.y1 = Parameter(torch.cuda.FloatTensor([y1]))
        else:
            self.x0 = Variable(torch.Tensor([x0]))
            self.y0 = Variable(torch.Tensor([y0]))
            self.x1 = Parameter(torch.Tensor([x1]))
            self.y1 = Parameter(torch.Tensor([y1]))
        self.imsize = imsize
        self.fuzz = fuzz
        self.use_cuda = use_cuda

    def forward(self):
        template = draw_line(self.x0, self.y0, self.x1, self.y1,
                             imsize=self.imsize, fuzz=self.fuzz,
                             use_cuda=self.use_cuda)
        template = torch.unsqueeze(template, dim=0)
        template = torch.unsqueeze(template, dim=0)

        return template


class BresenhamRenderNet(object):
    """Non-differentiable renderer. After we learn the parameters
    we should use this to render the final image so that it will
    look cleaner.
    """
    def __init__(self, x_list, y_list, pen_list=None, 
                 imsize=224, linewidth=1):
        super(BresenhamRenderNet, self).__init__()
        assert len(x_list) == len(y_list)
        assert len(x_list) == len(pen_list)
        self.n_points = len(x_list)
        self.x_list = x_list
        self.y_list = y_list
        if pen_list is None:
            # if none is provided, draw everything.
            pen_list = [2 for i in xrange(self.n_points)]
        self.pen_list = pen_list
        self.imsize = imsize
        if linewidth % 2 == 0:
            linewidth += 1
        self.linewidth = linewidth

    def forward(self):
        template = torch.zeros(self.imsize, self.imsize)
        for i in range(1, self.n_points):
            if self.pen_list[i] == 2:
                _template = draw_binary_line(int(round(self.x_list[i - 1])), int(round(self.y_list[i - 1])),
                                             int(round(self.x_list[i])), int(round(self.y_list[i])),
                                             imsize=self.imsize, width=self.linewidth)
            template += _template
        template = torch.clamp(template, 0, 1)
        template = torch.unsqueeze(template, dim=0)
        template = torch.unsqueeze(template, dim=0)

        return template


def exponential_smooth_min(A, dim=0, k=10):
    res = torch.sum(torch.exp(-k * A), dim=dim)
    return -torch.log(res / len(A)) / k


def draw_line(x0, y0, x1, y1, imsize=224, fuzz=1.0, use_cuda=False):
    """Given 2 points, populate a matrix with a smooth line from
    (x0, y0) to (x1, y1).

    :param x0: PyTorch Variable or Parameter
    :param y0: PyTorch Variable or Parameter
    :param x1: PyTorch Variable or Parameter
    :param y1: PyTorch Variable or Parameter
    :param imsize: size of matrix
    :param fuzz: amount of blurring
    :param use_cuda: create variables with cuda
    :return template: matrix with line segment on it
    """
    x0 = x0.repeat(imsize * imsize)
    y0 = y0.repeat(imsize * imsize)
    x1 = x1.repeat(imsize * imsize)
    y1 = y1.repeat(imsize * imsize)
    xp0 = Variable(torch.arange(0, imsize).repeat(imsize))
    if use_cuda:
        xp0 = xp0.cuda()
    yp0 = torch.t(xp0.view(imsize, imsize)).contiguous().view(-1)

    # if x1 is equal to x0, we can't calculate slope so we need to handle
    # this case separately
    ii_nonzero = x1 != x0
    ii_zero = torch.eq(x1, x0)
    n_zero = sum(ii_zero.data)

    if n_zero == 0:
        xp1, yp1 = gen_closest_point_on_line(x0[ii_nonzero], y0[ii_nonzero],
                                             x1[ii_nonzero], y1[ii_nonzero],
                                             xp0[ii_nonzero], yp0[ii_nonzero])
    else:  # this is a vertical line
        xp1, yp1 = gen_closest_point_on_vertical_line(x0[ii_zero], y0[ii_zero],
                                                      x1[ii_zero], y1[ii_zero],
                                                      xp0[ii_zero], yp0[ii_zero])

    # points may be out of the line segments range
    xp1 = torch.clamp(xp1, min=min((x0.data[0], x1.data[0])),
                      max=max((x0.data[0], x1.data[0])))
    yp1 = torch.clamp(yp1, min=min((y0.data[0], y1.data[0])),
                      max=max((y0.data[0], y1.data[0])))

    d = gen_euclidean_distance(xp0, yp0, xp1, yp1)
    d = torch.pow(d, fuzz)  # scale the differences
    template = d.view(imsize, imsize)
    return template


def draw_binary_line(x0, y0, x1, y1, imsize=224, width=1):
    """Non-differentiable way to draw a line with no fuzz

    :param x0: int, x coordinate of point 0
    :param y0: int, y coordinate of point 0
    :param x1: int, x coordinate of point 1
    :param y1: int, y coordinate of point 1
    :param imsize: size of image frame
    :param width: width of line
    :return template: torch Tensor of imsize x imsize
    """
    if width % 2 == 0:
        width += 1
    hw = int((width - 1) / 2)

    template = torch.zeros((imsize, imsize))
    dx, dy = x1 - x0, y1 - y0
    is_steep = abs(dy) > abs(dx)

    if is_steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    swapped = False
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        swapped = True

    dx = x1 - x0
    dy = y1 - y0

    error = int(dx / 2.0)
    ystep = 1 if y0 < y1 else  -1

    y = y0
    for x in range(x0, x1 + 1):
        if is_steep:
            if hw > 0:
                template[max(y-hw, 0):min(y+hw, imsize), 
                         max(x-hw, 0):min(x+hw, imsize)] = 1
            else:
                template[y, x] = 1
        else:
            if hw > 0:
                template[max(x-hw, 0):min(x+hw, imsize), 
                         max(y-hw, 0):min(y+hw, imsize)] = 1
            else:
                template[x, y] = 1
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    return template


def _gen_closest_point_on_line(x0, y0, x1, y1, xp, yp, eps=1e-10):
    """Same as gen_closest_point_on_line but here we make the
    assumption that x0 != x1 for all indexes.
    """
    n = (x1-x0)*yp*(y1-y0)+(y0-y1)*(y0*x1-x0*y1)+xp*torch.pow(x0-x1,2)
    d = torch.pow(x0-x1,2) + torch.pow(y0-y1,2)
    x = n/(d+eps)
    y = (y1-y0)/(x1-x0+eps)*x+(y0*x1-x0*y1)/(x1-x0+eps)
    return x, y


def gen_closest_point_on_vertical_line(x0, y0, x1, y1, xp, yp):
    return x1, yp


def gen_closest_point_on_line(x0, y0, x1, y1, xp, yp):
    """(x0, y0), (x1, y1) define a line. We want find the closest point
    on the line from (xp, yp). This only supports vectorized computation,
    meaning you have to pass Torch Tensors as points.

    :param x0: Torch tensor 1D of x coordinates of a point on the line
    :param y0: Torch tensor 1D of y coordinate of a point on the line
    :param x1: Torch tensor 1D of x coordinate of another point on the line
    :param y1: Torch tensor 1D of y coordinate of another point on the line
    :param xp: Torch tensor 1D of x coordinate of a point OFF the line
    :param yp: Torch tensor 1D of y coordinate of a point OFF the line
    :return x: Torch tensor 1D of x coordinate of the closest point on the line to (xp, yp)
    :return y: Torch tensor 1D of y coordinate of the closest point on the line to (xp, yp)
    """
    x, y = _gen_closest_point_on_line(x0, y0, x1, y1, xp, yp)
    return x, y


def gen_euclidean_distance(x0, y0, x1, y1, eps=1e-10):
    """Calculate Euclidean distance between (x0, y0) and (x1, y1).
    This only supports vectorized computation.

    :param x0: Torch tensor 1D of x coordinates of a point
    :param y0: Torch tensor 1D of y coordinate of a point
    :param x1: Torch tensor 1D of x coordinate of another point
    :param y1: Torch tensor 1D of y coordinate of another point
    """
    return torch.pow(torch.pow(x1 - x0, 2) + torch.pow(y1 - y0, 2) + eps, 0.5)
