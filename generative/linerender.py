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


class RenderNet(nn.Module):
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
        super(RenderNet, self).__init__()
        self.x0 = Variable(torch.Tensor([x0]))
        self.y0 = Variable(torch.Tensor([y0]))
        self.x1 = Parameter(torch.Tensor([x1]))
        self.y1 = Parameter(torch.Tensor([y1]))
        self.imsize = imsize
        self.fuzz = fuzz

    def forward(self):
        x0 = self.x0.repeat(self.imsize * self.imsize)
        y0 = self.y0.repeat(self.imsize * self.imsize)
        x1 = self.x1.repeat(self.imsize * self.imsize)
        y1 = self.y1.repeat(self.imsize * self.imsize)
        xp0 = Variable(torch.arange(0, self.imsize).repeat(self.imsize))
        yp0 = torch.t(xp0.view(self.imsize, self.imsize)).contiguous().view(-1)

        # if x1 is equal to x0, we can't calculate slope so we need to handle
        # this case separately
        ii_nonzero = x1 != x0
        ii_zero = torch.eq(x1, x0)
        n_zero = torch.sum(ii_zero).data[0]

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
        d = torch.pow(d, self.fuzz)  # scale the differences
        template = d.view(self.imsize, self.imsize)

        # renorm to 0 and 1
        tmin = torch.min(template)
        tmax = torch.max(template)
        template = (template - tmin) / (tmax - tmin)
        template = torch.unsqueeze(template, dim=0)
        template = torch.unsqueeze(template, dim=0)

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
