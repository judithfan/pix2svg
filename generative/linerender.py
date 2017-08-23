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
    such that it is differentiable. This is based on a
    simplified Bresenham's line algorithm: calculate slope and
    round to stay close to the slope.

    :param x0: fixed starting x coordinate
    :param y0: fixed starting y coordinate
    :param x1: initialization for ending x coordinate
    :param y1: initialization for ending y coordinate
    :param imsize: image size to generate
    """
    def __init__(self, x0, y0, x1, y1, imsize=224, linewidth=7):
        super(RenderNet, self).__init__()
        self.x0 = Variable(torch.Tensor([x0]))
        self.y0 = Variable(torch.Tensor([y0]))
        self.x1 = Parameter(torch.Tensor([x1]))
        self.y1 = Parameter(torch.Tensor([y1]))
        self.imsize = imsize
        if linewidth % 2 == 0:
            linewidth += 1
        assert linewidth > 1, 'Thin lines are not differentiable.'
        self.linewidth = linewidth

    def gen_kernel(self):
        linewidth = self.linewidth
        center = linewidth // 2
        kernel = torch.zeros((1, 1, linewidth, linewidth))
        # compose our kernel out of smaller kernels
        for i in range(center + 1):
            width = linewidth - i * 2
            value = (1.0 / (center + 1)) * (i + 1)
            _kernel = torch.ones((1, 1, width, width)) * value
            offset = (linewidth - width) // 2
            kernel[
                :,
                :,
                offset:linewidth - offset,
                offset:linewidth - offset,
            ] = _kernel
        return Variable(kernel)

    def smooth_index(self, template, kernel, x, y, _x, _y, value):
        """Indexing using Variable, even with scatter_ is not
        differentiable. This is our hacky (& slower) workaround.

        :param template: torch matrix (imsize x imsize)
        :param kernel: torch matrix (dim x dim x kH x kW)
        :param x: torch Parameter for x coordinate
        :param y: torch Parameter for y coordinate
        :param _x: integer for x coordinate
        :param _y: integer for y coordinate
        :param value: float value to set it to
        :return: None, in-place operation on template
        """
        z = x + y
        mask = z.expand(1, 1, self.imsize, self.imsize)
        difference = Variable(torch.zeros(1, 1, self.imsize, self.imsize))
        difference[:, :, _x, _y] = -z
        mask = z - torch.add(mask, difference)
        mask = torch.mul(mask, value / z)
        mask = F.conv2d(mask, kernel, padding=self.linewidth // 2)
        # convolve before adding
        template = torch.add(template, mask)
        # clip at value
        template = template.clamp(0, value)
        return template

    def forward(self):
        imsize = self.imsize
        linewidth = self.linewidth

        x0, y0 = self.x0, self.y0
        x1, y1 = self.x1, self.y1
        _x0, _y0 = int(x0.data[0]), int(y0.data[0])
        _x1, _y1 = int(x1.data[0]), int(y1.data[0])

        kernel = self.gen_kernel()  # to make smooth
        template = Variable(torch.zeros(1, 1, imsize, imsize))

        # start bresenham's line algorithm
        steep = False
        sx = 1 if torch.gt(x1, x0).data[0] > 0 else -1
        sy = 1 if torch.gt(y1, y0).data[0] > 0 else -1
        dx = torch.abs(x1 - x0)
        dy = torch.abs(y1 - y0)

        if torch.gt(dy, dx).data[0]:
            steep = True
            x0, y0 = y0, x0
            dx, dy = dy, dx
            sx, sy = sy, sx

        d = (2 * dy) - dx
        for i in range(0, int(dx.data[0])):
            if steep:
                template = self.smooth_index(template, kernel,
                                             y0, x0, _y0, _x0, 1)
            else:
                template = self.smooth_index(template, kernel,
                                             x0, y0, _x0, _y0, 1)
            while d.data[0] >= 0:
                y0 += sy
                d -= (2 * dx)
            x0 += sx
            d += (2 * dy)

        template = self.smooth_index(template, kernel,
                                     x1, y1, _x1, _y1, 1)
        template = torch.cat([template, template, template], dim=1)

        return template
