from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

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
    :param imsize: image size to generate
    :param fuzz: hyperparameter to scale differences; fuzz > 1 would
                 localize around the line; fuzz < 1 would make things
                 more uniform.
    :return template: imsize by imsize rendered sketch
    """
    def __init__(self, x_list, y_list, imsize=224, fuzz=1):
        super(SketchRenderNet, self).__init__()
        assert len(x_list) == len(y_list)
        self.n_points = len(x_list)
        self.x_list = [Parameter(torch.Tensor([x])) for x in x_list]
        self.y_list = [Parameter(torch.Tensor([y])) for y in y_list]
        self.imsize = imsize
        self.fuzz = fuzz

    def forward(self):
        template = Variable(torch.zeroes(self.imsize, self.imsize))
        for i in range(1, self.n_points):
            _template = draw_line(self.x_list[i - 1], self.y_list[i - 1],
                                  self.x_list[i], self.y_list[i],
                                  imsize=self.imsize, fuzz=self.fuzz)
            template += _template

        # renorm to 0 and 1
        tmin = torch.min(template)
        tmax = torch.max(template)
        template = (template - tmin) / (tmax - tmin)
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
    :return template: imsize by imsize rendered sketch
    """
    def __init__(self, x0, y0, x1, y1, imsize=224, fuzz=1):
        super(LineRenderNet, self).__init__()
        self.x0 = Variable(torch.Tensor([x0]))
        self.y0 = Variable(torch.Tensor([y0]))
        self.x1 = Parameter(torch.Tensor([x1]))
        self.y1 = Parameter(torch.Tensor([y1]))
        self.imsize = imsize
        self.fuzz = fuzz

    def forward(self):
        template = draw_line(self.x0, self.y0, self.x1, self.y1,
                             imsize=self.imsize, fuzz=self.fuzz)
        # renorm to 0 and 1
        tmin = torch.min(template)
        tmax = torch.max(template)
        template = (template - tmin) / (tmax - tmin)
        template = torch.unsqueeze(template, dim=0)
        template = torch.unsqueeze(template, dim=0)

        return template


def draw_line(x0, y0, x1, y1, imsize=224, fuzz=1.0):
    """Given 2 points, populate a matrix with a smooth line from
    (x0, y0) to (x1, y1).

    :param x0: PyTorch Variable or Parameter
    :param y0: PyTorch Variable or Parameter
    :param x1: PyTorch Variable or Parameter
    :param y1: PyTorch Variable or Parameter
    :param imsize: size of matrix
    :param fuzz: amount of blurring
    :return template: matrix with line segment on it
    """
    x0 = x0.repeat(imsize * imsize)
    y0 = y0.repeat(imsize * imsize)
    x1 = x1.repeat(imsize * imsize)
    y1 = y1.repeat(imsize * imsize)
    xp0 = Variable(torch.arange(0, imsize).repeat(imsize))
    yp0 = torch.t(xp0.view(imsize, imsize)).contiguous().view(-1)

    # if x1 is equal to x0, we can't calculate slope so we need to handle
    # this case separately
    ii_nonzero = x1 != x0
    ii_zero = torch.eq(x1, x0)
    n_zero = torch.sum(ii_zero).data[0]

    if not n_zero:
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
