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
        # save raw values for indexing
        self._x1 = x1
        self._y1 = y1
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

    def smooth_index(self, template, x, y, _x, _y):
        """Indexing using Variable, even with scatter_ is not
        differentiable. This is our hacky (& slower) workaround.

        :param template: torch matrix (imsize x imsize)
        :param x: torch Parameter for x coordinate
        :param y: torch Parameter for y coordinate
        :param _x: integer for x coordinate
        :param _y: integer for y coordinate
        :return: None, in-place operation on template
        """
        z = x + y
        mask = z.expand(self.imsize, self.imsize)
        difference = Variable(torch.zeros(self.imsize, self.imsize))
        difference[_x, _y] = -z
        mask = z - torch.add(mask, difference)
        mask = torch.mul(mask, 1 / z)
        template = torch.add(template, mask)
        return template

    def forward(self):
        imsize = self.imsize
        linewidth = self.linewidth

        x0, y0 = self.x0, self.y0
        x1, y1 = self.x1, self.y1
        _x1, _y1 = self._x1, self._y1
        x1.register_hook(print)
        y1.register_hook(print)

        # to be differentiable, make 1d first
        template = Variable(torch.zeros(imsize, imsize))
        template = self.smooth_index(template, x1, y1, _x1, _y1)

        # start bresenham's line algorithm
        # steep = False
        # sx = 1 if torch.gt(x1, x0).data[0] > 0 else -1
        # sy = 1 if torch.gt(y1, y0).data[0] > 0 else -1
        # dx = torch.abs(x1 - x0)
        # dy = torch.abs(y1 - y0)

        # if torch.gt(dy, dx).data[0]:
        #     steep = True
        #     x0, y0 = y0, x0
        #     dx, dy = dy, dx
        #     sx, sy = sy, sx

        # d = (2 * dy) - dx
        # for i in range(0, int(dx.data[0])):
        #     idx = mat2vecidx(y0, x0) if steep else mat2vecidx(x0, y0)
        #     idx = idx.long()
        #     template[idx] = 1
        #     while d.data[0] >= 0:
        #         y0 += sy
        #         d -= (2 * dx)
        #     x0 += sx
        #     d += (2 * dy)
        # idx = mat2vecidx(x1, y1).long()
        # template[idx] = 1
        # template.register_hook(print)

        # # reshape into matrix
        template = torch.unsqueeze(template, dim=0)  # add 3rd dim
        template = torch.cat([template, template, template], dim=0)  # concat
        template = torch.unsqueeze(template, dim=0)  # add 4th dim
        # # convolve against kernel to smooth
        # template = F.conv2d(template, kernel, padding=linewidth // 2)

        # # normalize the matrix to be from 0 and 1
        template_min = torch.min(template).expand_as(template)
        template_max = torch.max(template).expand_as(template)
        template = (template - template_min) / (template_max - template_min)

        return template
