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
    def __init__(self, x0, y0, x1, y1, imsize=256, linewidth=1):
        super(RenderNet, self).__init__()
        self.x0 = Variable(torch.Tensor([x0]), requires_grad=False)
        self.y0 = Variable(torch.Tensor([y0]), requires_grad=False)
        self.x1 = Parameter(torch.Tensor([x1]))
        self.y1 = Parameter(torch.Tensor([y1]))
        self.imsize = imsize
        self.linewidth = linewidth

    def gen_kernel(self):
        linewidth = self.linewidth
        center = linewidth // 2
        kernel = torch.zeros((3, 3, linewidth, linewidth))
        # compose our kernel out of smaller kernels
        for i in range(center + 1):
            width = linewidth - i * 2
            value = (1.0 / (center + 1)) * (i + 1)
            _kernel = torch.ones((width, width)) * value

            offset = (linewidth - width) // 2
            kernel[
                :,
                :,
                offset:linewidth - offset,
                offset:linewidth - offset,
            ] = _kernel
        return Variable(kernel)

    def forward(self):
        # render the points as a CHW Tensor
        xmin = int(torch.min(self.x0, self.x1).data[0])
        xmax = int(torch.max(self.x0, self.x1).data[0])
        ymin = int(torch.min(self.y0, self.y1).data[0])
        ymax = int(torch.max(self.y0, self.y1).data[0])
        slope = (ymax - ymin) / (xmax - xmin)

        padding = self.linewidth - 1
        templates = Variable(torch.ones((3, self.imsize + padding,
                                         self.imsize + padding)))
        for i in range(xmin, xmax + 1):
            x = i + padding
            y = int(round(x * slope + ymin)) + padding
            templates[:, x, y] = 0

        kernel = self.gen_kernel()
        templates = F.conv2d(torch.unsqueeze(templates, dim=0), kernel)

        template_min = torch.min(templates)
        template_max = torch.max(templates)
        templates = (templates - template_min) / (template_max - template_min)

        return templates
