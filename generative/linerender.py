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
                 fuzz=1, smoothness=8, use_cuda=False):
        super(SketchRenderNet, self).__init__()
        n_points = len(x_list)
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        
        if pen_list is None:
            # if none is provided, draw everything.
            pen_list = [2 for i in xrange(n_points)]

        # our parameters are the points of the sketch
        self.x_params = Parameter(torch.Tensor(x_list).type(dtype))
        self.y_params = Parameter(torch.Tensor(y_list).type(dtype))
        self.pen_params = torch.Tensor(pen_list).type(dtype)

        # save parameters
        self.imsize = imsize
        self.fuzz = fuzz
        self.smoothness = smoothness
        self.use_cuda = use_cuda
        

    def forward(self): 
        x0 = self.x_params[:-1]
        y0 = self.y_params[:-1]
        p0 = self.pen_params[:-1]
        x1 = self.x_params[1:]
        y1 = self.y_params[1:]
        p1 = self.pen_params[1:]

        template = draw_lines(x0, y0, x1, y1, p0, p1, imsize=self.imsize, 
                              use_cuda=self.use_cuda)
        template = torch.pow(template, self.fuzz)
        template = exponential_smooth_min(template, dim=0, k=self.smoothness)
        # add a dimension for batches and a dimension for channels
        template = torch.unsqueeze(template, dim=0)
        template = torch.unsqueeze(template, dim=0)
        
        return template


def draw_lines(x0, y0, x1, y1, p0, p1, imsize=224, use_cuda=False):
    """Given 2 points, populate a matrix with a smooth line from
    (x0, y0) to (x1, y1). This is vectorized across points in a 
    grid and for all lines.

    :param x0: PyTorch Variable or Parameter
    :param y0: PyTorch Variable or Parameter
    :param x1: PyTorch Variable or Parameter
    :param y1: PyTorch Variable or Parameter
    :param p0: pen strokes for (x0, y0)
    :param p1: pen strokes for (x1, y1)
    :param imsize: size of matrix
    :param use_cuda: create variables with cuda
    :return template: matrix with line segment on it
    """
    x0 = x0.unsqueeze(1).repeat(1, imsize * imsize)
    y0 = y0.unsqueeze(1).repeat(1, imsize * imsize)
    x1 = x1.unsqueeze(1).repeat(1, imsize * imsize)
    y1 = y1.unsqueeze(1).repeat(1, imsize * imsize)

    xp0 = Variable(torch.arange(0, imsize).repeat(imsize).type(dtype))
    yp0 = torch.t(xp0.view(imsize, imsize)).contiguous().view(-1)
    xp0 = xp0.unsqueeze(0)
    yp0 = yp0.unsqueeze(0)

    ii_vert = torch.eq(x1, x0)
    ii_nonvert = torch.ne(x1, x0)
    n_vert = int(torch.sum(ii_vert[:, 0].data))
    n_points = x0.size()[0]
    n_draw = sum(p1 == 2)

    # there are 2 scenarios: vertical lines and not-vertical
    # lines. we need to handle vertical lines separately b/c
    # we can't calculate slope. we reorder our points to be 
    # non-vert first, then vertical
    x0_0 = x0[ii_nonvert].view(-1, imsize * imsize)
    y0_0 = y0[ii_nonvert].view(-1, imsize * imsize)
    p0_0 = p0[ii_nonvert.data[:, 0]]
    x1_0 = x1[ii_nonvert].view(-1, imsize * imsize)
    y1_0 = y1[ii_nonvert].view(-1, imsize * imsize)
    p1_0 = p1[ii_nonvert.data[:, 0]]

    if n_vert > 0:
        x0_1 = x0[ii_vert].view(-1, imsize * imsize)
        y0_1 = y0[ii_vert].view(-1, imsize * imsize)
        p0_1 = p0[ii_vert.data[:, 0]]
        x1_1 = x1[ii_vert].view(-1, imsize * imsize)
        y1_1 = y1[ii_vert].view(-1, imsize * imsize)
        p1_1 = p1[ii_vert.data[:, 0]]
        x0 = torch.cat((x0_0, x0_1))
        y0 = torch.cat((y0_0, y0_1))
        x1 = torch.cat((x1_0, x1_1))
        y1 = torch.cat((y1_0, y1_1))
        p0 = torch.cat((p0_0, p0_1))
        p1 = torch.cat((p1_0, p1_1))
    else:
        x0 = x0_0
        y0 = y0_0
        x1 = x1_0
        y1 = y1_0
        p0 = p0_0
        p1 = p1_0

    # compute closest point on line for each of 2 scenarios
    xp1_0, yp1_0 = gen_closest_point_on_line(x0[:n_points - n_vert], y0_0[:n_points - n_vert], 
                                             x1_0[:n_points - n_vert], y1_0[:n_points - n_vert], 
                                             xp0, yp0)

    if n_vert > 0:
        x0_1 = x0[-n_vert:].view(-1, imsize * imsize)
        y0_1 = y0[-n_vert:].view(-1, imsize * imsize)
        x1_1 = x1[-n_vert:].view(-1, imsize * imsize)
        y1_1 = y1[-n_vert:].view(-1, imsize * imsize)
        xp1_1, yp1_1 = x1_1, yp0.repeat(n_vert, 1)

        xp1 = torch.cat((xp1_0, xp1_1), dim=0)
        yp1 = torch.cat((yp1_0, yp1_1), dim=0)
    else:
        xp1 = xp1_0
        yp1 = yp1_0

    # make sure no points exceed line segment endpoints
    xp1_min = torch.min(torch.stack((x0[:, 0], x1[:, 0])), dim=0)[0].data
    xp1_max = torch.max(torch.stack((x0[:, 0], x1[:, 0])), dim=0)[0].data
    yp1_min = torch.min(torch.stack((y0[:, 0], y1[:, 0])), dim=0)[0].data
    yp1_max = torch.max(torch.stack((y0[:, 0], y1[:, 0])), dim=0)[0].data

    xp1 = torch.stack([xp1[i, :].clamp(xp1_min[i], xp1_max[i]) 
                       for i in range(xp1.size()[0]) if p1[i] == 2])
    yp1 = torch.stack([yp1[i, :].clamp(yp1_min[i], yp1_max[i]) 
                       for i in range(yp1.size()[0]) if p1[i] == 2])

    # calculate euclidean distance
    d = gen_euclidean_distance(xp0, yp0, xp1, yp1)
    template = d.resize_as(xp1)
    template = template.view(n_draw, imsize, imsize)

    return template


def gen_closest_point_on_line(x0, y0, x1, y1, xp, yp, eps=1e-10):
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
    n = (x1-x0)*yp*(y1-y0)+(y0-y1)*(y0*x1-x0*y1)+xp*torch.pow(x0-x1,2)
    d = torch.pow(x0-x1,2) + torch.pow(y0-y1,2)
    x = n/(d+eps)
    y = (y1-y0)/(x1-x0+eps)*x+(y0*x1-x0*y1)/(x1-x0+eps)
    return x, y


def exponential_smooth_min(A, dim=0, k=32):
    A_max = torch.max(A, dim=dim)[0]
    B = torch.sum(torch.exp(-k * (A - A_max)), dim=dim)
    N = k * A_max
    C = -torch.log(B) + N
    return C / k


def gen_euclidean_distance(x0, y0, x1, y1, eps=1e-10):
    """Calculate Euclidean distance between (x0, y0) and (x1, y1).
    This only supports vectorized computation.

    :param x0: Torch tensor 1D of x coordinates of a point
    :param y0: Torch tensor 1D of y coordinate of a point
    :param x1: Torch tensor 1D of x coordinate of another point
    :param y1: Torch tensor 1D of y coordinate of another point
    """
    return torch.pow(torch.pow(x1 - x0, 2) + torch.pow(y1 - y0, 2) + eps, 0.5)



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
