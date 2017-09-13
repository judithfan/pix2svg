from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import sys
import copy
import numpy as np


import torch
import torch.nn as nn


class SplineRenderNet(nn.Module):
    """Renders an image of a spline as a CxHxW matrix given 3 points
    such that it is differentiable. The intensity of each
    pixel is the shortest distance from each pixel to the spline. 
    Only parabolic splines are supported... b/c I can't closed-form
    factor a 4th degree polynomial.

    :param x0: fixed starting x coordinate
    :param y0: fixed starting y coordinate
    :param x1: initialization for tangent x coordinate
    :param y1: initialization for tangent y coordinate
    :param x2: initialization for ending x coordinate
    :param y2: initialization for ending y coordinate
    :param imsize: image size to generate
    :param fuzz: hyperparameter to scale differences; fuzz > 1 would
                 localize around the line; fuzz < 1 would make things
                 more uniform.
    :param use_cuda: make variables using cuda
    :return template: imsize by imsize rendered sketch
    """
    def __init__(self, x0, y0, x1, y1, x2, y2, imsize=224, fuzz=1, use_cuda=False):
        super(SplineRenderNet, self).__init__()
        if use_cuda:
            self.x0 = Variable(torch.cuda.FloatTensor([x0]))
            self.y0 = Variable(torch.cuda.FloatTensor([y0]))
            self.x1 = Parameter(torch.cuda.FloatTensor([x1]))
            self.y1 = Parameter(torch.cuda.FloatTensor([y1]))
            self.x2 = Parameter(torch.cuda.FloatTensor([x2]))
            self.y2 = Parameter(torch.cuda.FloatTensor([y2]))
        else:
            self.x0 = Variable(torch.FloatTensor([x0]))
            self.y0 = Variable(torch.FloatTensor([y0]))
            self.x1 = Parameter(torch.FloatTensor([x1]))
            self.y1 = Parameter(torch.FloatTensor([y1]))
            self.x1 = Parameter(torch.FloatTensor([x2]))
            self.y1 = Parameter(torch.FloatTensor([y2]))
        self.imsize = imsize
        self.fuzz = fuzz
        self.use_cuda = use_cuda

    def forward(self):
        template = draw_spline(self.x0, self.y0, self.x1, self.y1, self.x2, self.y2,
                               imsize=self.imsize, fuzz=self.fuzz,
                               use_cuda=self.use_cuda)
        template = torch.unsqueeze(template, dim=0)
        template = torch.unsqueeze(template, dim=0)

        return template


class ParabolicBrezierRenderNet(object):
    """Non-differentiable spline renderer using polynomials of degree 2. 
    After we learn the parameters we should use this to render the final 
    image so that it will look cleaner.
    """
    def __init__(self, x0_list, y0_list, x1_list, y1_list,
                 x2_list, y2_list,  imsize=224, linewidth=1, tstep=0.01):
        super(ParabolicBrezierRenderNet, self).__init__()
        assert len(x0_list) == len(y0_list)
        assert len(x1_list) == len(y1_list)
        assert len(x2_list) == len(y2_list)
        assert len(x0_list) == len(x1_list)
        assert len(x0_list) == len(x2_list)
        assert tstep > 0 and tstep < 1
        self.n_points = len(x0_list)
        self.x0_list = x0_list
        self.y0_list = y0_list
        self.x1_list = x1_list
        self.y1_list = y1_list
        self.x2_list = x2_list
        self.y2_list = y2_list
        self.imsize = imsize
        if linewidth % 2 == 0:
            linewidth += 1
        self.linewidth = linewidth
        self.tstep = tstep

    def forward(self):
        template = torch.zeros(self.imsize, self.imsize)
        for i in range(self.n_points):
            x0 = self.x0_list[i]
            y0 = self.y0_list[i]
            x1 = self.x1_list[i]
            y1 = self.y1_list[i]
            x2 = self.x2_list[i]
            y2 = self.y2_list[i]
            _template = draw_binary_spline(x0, y0, x1, y1, x2, y2, imsize=self.imsize,
                                           width=self.linewidth, tstep=self.tstep)
            template += _template
        template = torch.clamp(template, 0, 1)
        template = torch.unsqueeze(template, dim=0)
        template = torch.unsqueeze(template, dim=0)

        return template


def draw_spline(x0, y0, x1, y1, x2, y2, imsize=224, fuzz=1.0, use_cuda=False):
    """Given 2 points, populate a matrix with a differentiable spline from
    (x0, y0) to (x2, y2) with parabolic tangent point (x1, y1).

    :param x0: PyTorch Variable or Parameter
    :param y0: PyTorch Variable or Parameter
    :param x1: PyTorch Variable or Parameter
    :param y1: PyTorch Variable or Parameter
    :param x2: PyTorch Variable or Parameter
    :param y2: PyTorch Variable or Parameter
    :param imsize: size of matrix
    :param fuzz: amount of blurring
    :param use_cuda: create variables with cuda
    :return template: matrix with line segment on it
    """
    xp0 = Variable(torch.arange(0, imsize).repeat(imsize)) 
    yp0 = torch.t(xp0.view(imsize, imsize)).contiguous().view(-1)
    if use_cuda:
        xp0 = xp0.cuda()
        yp0 = yp0.cuda()

    x0 = x0.repeat(imsize * imsize)
    y0 = y0.repeat(imsize * imsize)
    x1 = x1.repeat(imsize * imsize)
    y1 = y1.repeat(imsize * imsize)
    x2 = x2.repeat(imsize * imsize)
    y2 = y2.repeat(imsize * imsize)

    # unlike line, with splines we don't need to worry about x0 and x2
    # being along the same axis. we also don't need to worry about being
    # out of the splines range.
    xp1, yp1 = find_closest_point_on_spline(x0, y0, x1, y1, xp0, yp0)
    d = gen_euclidean_distance(xp0, yp0, xp1, yp1)
    d = torch.pow(d, fuzz)  # scale the differences
    template = d.view(imsize, imsize)
    return template


def find_closest_point_on_spline(x0, y0, x1, y1, x2, y2, xp, yp):
    """Finds the closest point on spline by looking at the derivative
    of the closest form derivative of the Brezier curve to get tangent
    slope and taking advantage of the fact that (xp, yp) should be 
    orthogonal to said point.
    
    This is a vectorized and PyTorch differentiable operation so all 
    inputs should be Torch Tensors.
    
    :param x0, y0: coordinates of endpoint 1 of spline
    :param x2, y2: coordinates of endpoint 2 of spline
    :param x1, y1: coordinates of tangent point
    :param xp, yp: coordinates of point that we want to find the closest
                   point for.
    """
    assert x0.size()[0] == y0.size()[0]
    assert x1.size()[0] == y1.size()[0]
    assert x2.size()[0] == y2.size()[0]
    assert x0.size()[0] == x1.size()[0]
    assert x1.size()[0] == x2.size()[0]
    n = x0.size()[0]
    
    # default position to (x0, y0) - (xp, yp)
    pos_x = x0 - xp
    pos_y = y0 - yp
    
    # search points of Bezier curve if (xp, yp) dot (dpoint / dt) = 0
    A_x = x1 - x0
    A_y = y1 - y0
    B_x = x0 - 2 * x1 + x2
    B_y = y0 - 2 * y1 + y2
    
    a = B_x * B_x + B_y * B_y
    b = 3 * (A_x * B_x + A_y * B_y)
    c = 2 * (A_x * A_x + A_y * A_y) + pos_x * B_x + pos_y * B_y
    d = pos_x * A_x + pos_y * A_y
    solution = solve_cubic_polynomial(a, b, c, d)
    ix_not_nan = 1 - is_nan(solution)
    
    d0 = euclidean_dist(xp, yp, x0, y0)
    d2 = euclidean_dist(xp, yp, x2, y2)
    
    # variables to track distances
    dist_min = torch.ones(n) * sys.maxint
    t_min = torch.ones(n) * sys.maxint
    pos_min_x = torch.ones(n) * np.nan
    pos_min_y = torch.ones(n) * np.nan
    try_endpoints = torch.zeros(n).byte()
    
    n_nan = torch.sum(ix_not_nan)
    if n_nan < n * 3:  # has at least 1 root
        # find closest point
        for i in range(3):
            if sum(ix_not_nan[:, i]) == 0:
                break

            root = solution[:, i]
            root_not_nan = root[ix_not_nan[:, i]]

            ix_ge_0 = root_not_nan >= 0
            ix_le_1 = root_not_nan <= 1
            ix_ge_not_nan_gt_0_le_1 = ix_ge_0 + ix_le_1 + ix_not_nan[:, i]
            ix_ge_not_nan_gt_0_le_1 = ix_ge_not_nan_gt_0_le_1 == 3
            
            if sum(ix_ge_not_nan_gt_0_le_1) > 0:
                root_ge_not_nan_gt_0_le_1 = root[ix_ge_not_nan_gt_0_le_1]
                pos_x, pos_y = get_position(x0, y0, x1, y1, x2, y2,
                                            root_ge_not_nan_gt_0_le_1)
                dist = euclidean_dist(xp, yp, pos_x, pos_y)
                
                ix_best = dist < dist_min[ix_ge_not_nan_gt_0_le_1]
                if sum(ix_best) > 0:
                    # ix_update is equiv. to ix_best but |ix_update| = |dist_min|
                    # where |ix_best| = |ix_ge_not_nan_gt_0_le_1|
                    ix_update = ix_ge_not_nan_gt_0_le_1
                    ix_update[ix_best] = 1
                    ix_update[1 - ix_best] = 0

                    dist_min[ix_update] = dist[ix_best]
                    t_min[ix_update] = root_ge_not_nan_gt_0_le_1[ix_best]
                    pos_min_x[ix_update] = pos_x[ix_best]
                    pos_min_y[ix_update] = pos_y[ix_best]
        
        ix_t_min_neq_maxint = t_min != sys.maxint
        ix_dist_min_lt_d0 = dist_min < d0
        ix_dist_min_lt_d2 = dist_min < d2
        ix_triplet = ix_t_min_neq_maxint + ix_dist_min_lt_d0 + ix_dist_min_lt_d2
        try_endpoints = try_endpoints + ix_triplet
        try_endpoints = 1 - torch.clamp(try_endpoints, 0, 1)
        # equals 1 if we want to try to find an endpoint
        
    if sum(try_endpoints) > 0:
        # if we've reached here, then closest point is on 2 end points
        ix_d0_lt_d2 = d0 < d2
        ix_d0_ge_d2 = 1 - ix_d0_lt_d2
        
        ix_try_d0_lt_d2 = try_endpoints + ix_d0_lt_d2
        ix_try_d0_lt_d2 = ix_try_d0_lt_d2 == 2
        
        if sum(ix_try_d0_lt_d2) > 0:
            dist_min[ix_try_d0_lt_d2] = d0[ix_try_d0_lt_d2]
            t_min[ix_try_d0_lt_d2] = 0
            pos_min_x[ix_try_d0_lt_d2] = x0[ix_try_d0_lt_d2]
            pos_min_y[ix_try_d0_lt_d2] = y0[ix_try_d0_lt_d2]

        ix_try_d0_ge_d2 = try_endpoints + ix_d0_ge_d2
        ix_try_d0_ge_d2 = ix_try_d0_ge_d2 == 2
            
        if sum(ix_try_d0_ge_d2) > 0:
            dist_min[ix_try_d0_ge_d2] = d2[ix_try_d0_ge_d2]
            t_min[ix_try_d0_ge_d2] = 1
            pos_min_x[ix_try_d0_ge_d2] = x2[ix_try_d0_ge_d2]
            pos_min_y[ix_try_d0_ge_d2] = y2[ix_try_d0_ge_d2]
    
    return (pos_min_x, pos_min_y)


def solve_cubic_polynomial(a, b, c, d, eps=0.0000001):
    """Solves cubic polynomial using Cardano's Formula. This also
    covers the case that a = 0, in which we solve a 2nd degree polynomial. 
    This is vectorized and differentiable so all inputs should b
    PyTorch Tensors/Variables. The lengths each of the vectors 
    must agree. 
    
    Given a polynomial: at^3 + bt^2 + ct + d = 0:
    
    :param a: cubic term
    :param b: 2nd degree term
    :param c: 1st degree term
    :param d: bias term
    :return roots: [...] (up to 3 roots)
                   imaginary roots will not be used.
    """
    assert a.size()[0] == b.size()[0]
    assert b.size()[0] == c.size()[0]
    assert c.size()[0] == d.size()[0]
    n = a.size()[0]  
    # store all roots across vector
    roots = torch.ones(n, 3) * np.nan
    ix_3 = torch.abs(a) > eps
    ix_2 = 1 - ix_3
    
    # handle the 3 polynomial cases
    if sum(ix_3) > 0:
        a_3, b_3, c_3, d_3 = a[ix_3], b[ix_3], c[ix_3], d[ix_3]
        roots_3 = _solve_cubic_polynomial(a_3, b_3, c_3, d_3, eps=eps)
        roots[:, 0][ix_3] = roots_3[:, 0]
        roots[:, 1][ix_3] = roots_3[:, 1]
        roots[:, 2][ix_3] = roots_3[:, 2]
    
    # handle the 2 polynomial cases
    if sum(ix_2) > 0:
        a_2, b_2, c_2, d_2 = a[ix_2], b[ix_2], c[ix_2], d[ix_2]
        roots_2 = _solve_parabolic_polynomial(a_2, b_2, c_2, d_2, eps=eps)
        roots[:, 0][ix_2] = roots_2[:, 0]
        roots[:, 1][ix_2] = roots_2[:, 1]
        roots[:, 2][ix_2] = roots_2[:, 2]
    
    return roots


def _solve_cubic_polynomial(a, b, c, d, eps=0.0000001):
    n = a.size()[0]  
    roots = torch.ones(n, 3) * np.nan
    z = copy.deepcopy(a)
    a = b / z
    b = c / z
    c = d / z
    # setup for Cardano's formula
    p = b - a * a / 3.
    q = a * (2 * a * a - 9 * b) / 27. + c
    p3 = p * p * p
    D = q * q + 4 * p3 / 27.
    offset = -a / 3.

    # if D > eps
    ix_D_gt_eps = D > eps
    if sum(ix_D_gt_eps) > 0:
        z_D_gt_eps = torch.sqrt(D[ix_D_gt_eps])
        u_D_gt_eps = (-q[ix_D_gt_eps] + z_D_gt_eps) / 2.
        v_D_gt_eps = (-q[ix_D_gt_eps] - z_D_gt_eps) / 2.

        # u = u**(1 / 3)  if (u >= 0) else -(-u)**(1 / 3)
        u_D_gt_eps_ge_0 = u_D_gt_eps >= 0
        tmp = u_D_gt_eps[u_D_gt_eps_ge_0]
        u_D_gt_eps[u_D_gt_eps_ge_0] = tmp**(1 / 3.)
        tmp = u_D_gt_eps[1 - u_D_gt_eps_ge_0]
        u_D_gt_eps[1 - u_D_gt_eps_ge_0] = -(-tmp)**(1 / 3.)

        # v = v**(1 / 3)  if (v >= 0) else -(-v)**(1 / 3)
        v_D_gt_eps_ge_0 = v_D_gt_eps >= 0
        tmp = v_D_gt_eps[v_D_gt_eps_ge_0]
        v_D_gt_eps[v_D_gt_eps_ge_0] = tmp**(1 / 3.)
        tmp = v_D_gt_eps[1 - v_D_gt_eps_ge_0]
        v_D_gt_eps[1 - v_D_gt_eps_ge_0] = -(-tmp)**(1 / 3.)

        root1 = u_D_gt_eps + v_D_gt_eps + offset[ix_D_gt_eps]
        roots[:, 0][ix_D_gt_eps] = root1

    # if D < -eps
    ix_D_lt_neps = D < -eps
    if sum(ix_D_lt_neps) > 0:
        u_D_lt_neps = 2 * torch.sqrt(-p[ix_D_lt_neps] / 3.)
        v_D_lt_neps = torch.acos(-torch.sqrt(-27. / p3[ix_D_lt_neps]) * q[ix_D_lt_neps] / 2.) / 3.
        root1 = u * torch.cos(v_D_lt_neps) + offset[ix_D_lt_neps]
        root2 = u * torch.cos(v_D_lt_neps + 2 * math.pi / 3.) + offset[ix_D_lt_neps]
        root3 = u * torch.cos(v_D_lt_neps + 4 * math.pi / 3.) + offset[ix_D_lt_neps]

        roots[:, 0][ix_D_lt_neps] = root1
        roots[:, 1][ix_D_lt_neps] = root2
        roots[:, 2][ix_D_lt_neps] = root3

    # if D <= eps and D >= -eps
    ix_D_le_eps = D <= eps
    ix_D_ge_neps = D >= -eps
    ix_D_near_0 = ix_D_le_eps + ix_D_ge_neps
    ix_D_near_0 = ix_D_near_0 == 2
    
    if sum(ix_D_near_0) > 0:
        q_D_near_0 = q[ix_D_near_0]
        u_D_near_0 = torch.zeros(q_D_near_0.size())
    
        ix_D_near_0_q_lt_0 = q_D_near_0 < 0
        u_D_near_0[ix_D_near_0_q_lt_0] = (-q_D_near_0[ix_D_near_0_q_lt_0] / 2.)**(1 / 3.)
        u_D_near_0[1 - ix_D_near_0_q_lt_0] = -(q_D_near_0[1 - ix_D_near_0_q_lt_0] / 2.)**(1 / 3.)

        root1 = 2 * u_D_near_0 + offset[ix_D_near_0]
        root2 = -u_D_near_0 + offset[ix_D_near_0]
        roots[:, 0][ix_D_near_0] = roots1
        roots[:, 1][ix_D_near_0] = roots2

    return roots


def _solve_parabolic_polynomial(a, b, c, d, eps=0.0000001):
    n = a.size()[0]  
    roots = torch.ones(n, 3) * np.nan
    
    # convert to at^2 + bt + c = 0
    a = b
    b = c
    c = d
    
    # handle case when a is non-positive and b is positive
    ix_a_le_eps = torch.abs(a) <= eps
    ix_b_gt_eps = torch.abs(b) > eps
    
    ix_a_le_eps_b_gt_eps = ix_a_le_eps + ix_b_gt_eps
    ix_a_le_eps_b_gt_eps = ix_a_le_eps_b_gt_eps == 2
    
    if sum(ix_a_le_eps_b_gt_eps) > 0:
        root = -c[ix_a_le_eps_b_gt_eps] / b[ix_a_le_eps_b_gt_eps]
        roots[:, 0][ix_a_le_eps_b_gt_eps] = root

    D = b*b - 4*a*c
    
    # handle case when D is gt eps
    ix_D_gt_eps = D > eps
    if sum(ix_D_gt_eps) > 0:
        root1 = (-b[ix_D_gt_eps] - torch.pow(D[ix_D_gt_eps], 0.5)) / (2. * a[ix_D_gt_eps])
        root2 = (-b[ix_D_gt_eps] + torch.pow(D[ix_D_gt_eps], 0.5)) / (2. * a[ix_D_gt_eps])
        roots[:, 0][ix_D_gt_eps] = root1
        roots[:, 1][ix_D_gt_eps] = root2
    
    # handle case when D is equal to 0
    ix_D_le_eps = D <= eps
    ix_D_ge_neps = D >= -eps
    ix_D_near_0 = ix_D_le_eps + ix_D_ge_neps
    ix_D_near_0 = ix_D_near_0 == 2
    
    if sum(ix_D_near_0) > 0:
        root = -b[ix_D_near_0] / (2. * a[ix_D_near_0])
        roots[:, 0][ix_D_near_0] = root
    
    return roots


def is_nan(x):
    return torch.ne(x, x)


def euclidean_dist(x0, y0, x1, y1):
    return torch.pow((x0 - x1)**2 + (y0 - y1)**2, 0.5)


def get_position(x0, y0, x1, y1, x2, y2, t):
    a = (1 - t)**2
    b = 2 * t * (1 - t)
    c = t**2
    pos_x = a * x0 + b * x1 + c * x2
    pos_y = a * y0 + b * y1 + c * y2
    return pos_x, pos_y


def draw_binary_spline(x0, y0, x1, y1, x2, y2, imsize=224, width=1, tstep=0.01):
    """Non-differentiable way to draw a spline with no fuzz

    :param x0: int, x coordinate of point 0
    :param y0: int, y coordinate of point 0
    :param x1: int, x coordinate of (tangent) point 1 
    :param y1: int, y coordinate of (tangent) point 1
    :param x2: int, x coordinate of point 2
    :param y2: int, y coordinate of point 2
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
    t = torch.t(t.repeat(2).view(-1, n_t))

    p_t = ((1-t)**2)*p0 + 2*(1-t)*t*p1 + (t**2)*p2
    p_t = torch.round(p_t).int()

    for i in range(n_t):
        x_t, y_t = p_t[i, 0], p_t[i, 1]
        if hw > 0:
            template[max(y_t-hw, 0):min(y_t+hw, imsize), 
                     max(x_t-hw, 0):min(x_t+hw, imsize)] = 1
        else:
            template[y_t, x_t] = 1

    return template

