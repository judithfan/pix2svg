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
        self.x0 = Variable(torch.Tensor([x0]), requires_grad=False)
        self.y0 = Variable(torch.Tensor([y0]), requires_grad=False)
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
        kernel = torch.zeros((3, 3, linewidth, linewidth))
        # compose our kernel out of smaller kernels
        for i in range(center + 1):
            width = linewidth - i * 2
            value = (1.0 / (center + 1)) * (i + 1)
            _kernel = torch.ones((3, 3, width, width)) * value

            offset = (linewidth - width) // 2
            kernel[
                :,
                :,
                offset:linewidth - offset,
                offset:linewidth - offset,
            ] = _kernel
        return Variable(kernel)

    def forward(self):
        # store image in templates
        templates = Variable(torch.zeros([3, self.imsize, self.imsize]))
        x0, y0 = self.x0, self.y0
        x1, y1 = self.x1, self.y1

        # start bresenham's line algorithm
        steep = False
        dx = torch.abs(x1 - x0)
        dy = torch.abs(y1 - y0)
        sx = 1 if torch.gt(x1, x0).data[0] > 0 else -1
        sy = 1 if torch.gt(y1, y0).data[0] > 0 else -1

        if torch.gt(dy, dx).data[0]:
            steep = True
            x0, y0 = y0, x0
            dx, dy = dy, dx
            sx, sy = sy, sx

        d = (2 * dy) - dx
        for i in range(0, int(dx.data[0])):
            if steep:
                templates[:, int(y0.data[0]), int(x0.data[0])] = 1
            else:
                templates[:, int(x0.data[0]), int(y0.data[0])] = 1
            while d.data[0] >= 0:
                y0 += sy
                d -= (2 * dx)
            x0 += sx
            d += (2 * dy)
        templates[:, int(x1.data[0]), int(y1.data[0])] = 1

        # convolve against kernel to smooth
        kernel = self.gen_kernel()
        templates = F.conv2d(torch.unsqueeze(templates, dim=0), kernel,
                             padding=self.linewidth // 2)

        template_min = torch.min(templates).expand_as(templates)
        template_max = torch.max(templates).expand_as(templates)
        templates = (templates - template_min) / (template_max - template_min)
        templates = 1 - torch.squeeze(templates, dim=0)
        return templates
