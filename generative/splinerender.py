from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch


class CubicBrezierRenderNet(object):
    """Non-differentiable spline renderer. After we learn the parameters
    we should use this to render the final image so that it will
    look cleaner.
    """
    def __init__(self, x_list, y_list, tan_x_list, tan_y_list, imsize=224, linewidth=1, tstep=0.01):
        super(CubicBrezierRenderNet, self).__init__()
        assert len(x_list) == len(y_list)
        assert len(tan_x_list) == len(tan_y_list)
        assert len(x_list) == len(tan_x_list)
        assert tstep > 0 and tstep < 1
        self.n_points = len(x_list)
        self.x_list = x_list
        self.y_list = y_list
        self.tan_x_list = tan_x_list
        self.tan_y_list = tan_y_list
        self.imsize = imsize
        if linewidth % 2 == 0:
            linewidth += 1
        self.linewidth = linewidth
        self.tstep = tstep

    def forward(self):
        template = torch.zeros(self.imsize, self.imsize)
        for i in range(1, self.n_points):
            x0 = self.x_list[i - 1]
            y0 = self.y_list[i - 1]
            x3 = self.x_list[i]
            y3 = self.y_list[i]
            x1 = self.tan_x_list[i - 1]
            y1 = self.tan_y_list[i - 1]
            x2 = self.tan_x_list[i]
            y2 = self.tan_y_list[i]

            _template = draw_binary_spline(x0, y0, x1, y1, x2, y2, x3, y3,
                                           imsize=self.imsize, width=self.linewidth,
                                           tstep=self.tstep)
            template += _template
        template = torch.clamp(template, 0, 1)
        template = torch.unsqueeze(template, dim=0)
        template = torch.unsqueeze(template, dim=0)

        return template


def draw_binary_spline(x0, y0, x1, y1, x2, y2, x3, y3, imsize=224, width=1, tstep=0.01):
    """Non-differentiable way to draw a spline with no fuzz

    :param x0: int, x coordinate of point 0
    :param y0: int, y coordinate of point 0
    :param x1: int, x coordinate of (tangent) point 1 
    :param y1: int, y coordinate of (tangent) point 1
    :param x2: int, x coordinate of (tangent) point 2
    :param y2: int, y coordinate of (tangent) point 2
    :param x3: int, x coordinate of point 3
    :param y3: int, y coordinate of point 3
    :param imsize: size of image frame
    :param width: width of line
    :return template: torch Tensor of imsize x imsize
    """

    if width % 2 == 0:
        width += 1
    hw = int((width - 1) / 2)

    # we will be populating this.
    template = torch.zeros((imsize, imsize))

    t = torch.arange(0, 1, tstep)
    n_t = t.size()[0]

    p0 = torch.Tensor([x0, y0]).repeat(n_t).view(n_t, -1)
    p1 = torch.Tensor([x1, y1]).repeat(n_t).view(n_t, -1)
    p2 = torch.Tensor([x2, y2]).repeat(n_t).view(n_t, -1)
    p3 = torch.Tensor([x3, y3]).repeat(n_t).view(n_t, -1)
    t = torch.t(t.repeat(2).view(-1, n_t))

    p_t = ((1-t)**3)*p0 + (3*(1-t)**2)*t*p1 + (3*(1-t)*t**2)*p2 + t**3*p3  
    p_t = torch.round(p_t).int()

    for i in range(n_t):
        x_t, y_t = p_t[i, 0], p_t[i, 1]
        if hw > 0:
            template[max(y_t-hw, 0):min(y_t+hw, imsize), 
                     max(x_t-hw, 0):min(x_t+hw, imsize)] = 1
        else:
            template[y_t, x_t] = 1

    return template

