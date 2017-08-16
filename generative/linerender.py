from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from PIL import Image


class DifferentiableLineRenderer(object):
    """Renders an image as a CxHxW matrix given 2 points
    such that it is differentiable. This is based on Bresenham's
    line algorithm.

    :param template: numpy array of (3 x imsize x imsize)
                     if none, start from tensor of zeros
    :param imsize: image size to generate
    """
    def __init__(self, template=None, imsize=256):
        if template is None:
            template = np.ones((3, imsize, imsize))*255
        self.template = template
        self.imsize = imsize

    def draw(self, x0, y0, x1, y1):
        assert x0 >= 0 and x0 < self.imsize
        assert y0 >= 0 and y0 < self.imsize
        assert x1 >= 0 and x1 < self.imsize
        assert y1 >= 0 and y1 < self.imsize

        template = self.template
        initial = True
        quit = False

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while not quit:
            if initial:
                initial = False
                template[:, x0, y0] = 0

            if x0 == x1 and y0 == y1:
                quit = True
                template[:, x1, y1] = 0

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            template[:, x0, y0] = 0

        self.template = np.uint8(template)
        return self.template

    def savefig(self, path):
        im = Image.fromarray(self.template.T)
        im.save(path)
