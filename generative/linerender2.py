from __future__ import division

import numpy as np
from numpy import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

class RenderLine(torch.autograd.Function):    
    '''
    inspired by Bresenham's algorithm (but not)
    Input: Takes starting point (p0) and endpoint (p1).
    Output: CURRENTLY: set of pixel coordinates // WANT: matrix with pixels filled in (see fill grid)
    TODO: add option to thicken lines, add option to gaussian blur
    '''
    def __init__(self):
        super(RenderLine,self).__init__()
        self.imsize = 250
        
    def forward(self,p0,p1):
        '''
        intuition: increment by x pixel, and fill in the closest y pixel as you go
        '''   
        # get pixel coordinates
        self.x0 = Variable(torch.Tensor([p0[0]]))
        self.y0 = Variable(torch.Tensor([p0[1]]))
        self.x1 = Variable(torch.Tensor([p1[0]]))
        self.y1 = Variable(torch.Tensor([p1[1]]))
        self.dx = torch.abs(self.x1-self.x0) 
        self.dy = torch.abs(self.y1-self.y0)
        self.slope = self.dy/self.dx
        self.xpix = torch.linspace(self.x0.data.numpy()[0].astype('int'),
                                   self.x1.data.numpy()[0].astype('int'),
                                   torch.abs(self.dx).data.numpy()[0].astype('int')+1)
        self.ypix = torch.stack([
            torch.round(torch.Tensor([_y])) 
            for i,_y in enumerate(torch.linspace(self.y0.data.numpy()[0].astype('int'),
                                                 self.y1.data.numpy()[0].astype('int'),
                                                 torch.abs(self.dx).data.numpy()[0].astype('int')+1))],dim=0)
        # self.xpix, self.ypix are floatTensors containing pixel coordinates
        return self.xpix,self.ypix
        # # now use to fill grid [this does NOT work right now]
        # mat = torch.zeros(self.imsize,self.imsize)
        # for _x,_y in zip(self.xpix,self.ypix):
        #     print type(_x)
        #     mat[int(_x),int(_y)] = 250                
        # return mat

