"""Pixel-wise loss against a sketch of a vertical line and a
generated sketch using beam search. The generated sketch
is given the ground truth anchor point (x0, y0). The test
confirms that via beam search, we can converge to the correct point.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys; sys.path.append('..')
import numpy as np

import torch
from torch.autograd import Variable

from pix2sketch import sample_endpoint_gaussian2d as sample_endpoint
from pix2sketch import (sample_action, pixel_sketch_loss)

from rendertest import gen_ground_truth
from linerender import RenderNet


class BeamSearch(object):
    def __init__(self, x0, y0, imsize, beam_width=2, n_samples=100, n_iters=10, stdev=2, fuzz=1.0):
        self.x_beam_queue = np.ones(beam_width) * x0
        self.y_beam_queue = np.ones(beam_width) * y0

        self.x_beam_paths = np.zeros((beam_width, n_iters + 1))
        self.y_beam_paths = np.zeros((beam_width, n_iters + 1))
        self.action_beam_paths = [[] for b in range(beam_width)]
        # init path with starting point
        self.x_beam_paths[:, 0] = x0
        self.y_beam_paths[:, 0] = y0

        # save parameters
        self.beam_width = beam_width
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.stdev = stdev
        self.imsize = imsize
        self.fuzz = fuzz

        self.beam_sketches = Variable(torch.zeros((beam_width, 1, 11, 11)))

    def train(self, gt_sketch, epoch):
        for b in range(self.beam_width):
            samples = sample_endpoint(self.x_beam_queue[b], self.y_beam_queue[b],
                                      std=self.stdev, size=self.n_samples,
                                      min_x=0, max_x=self.imsize, min_y=0, max_y=self.imsize)
            x_samples, y_samples = samples[:, 0], samples[:, 1]
            action_sample = sample_action()

            sketches = Variable(torch.zeros((self.n_samples, 1, self.imsize, self.imsize)))
            for i in range(self.n_samples):
                action_path = self.action_beam_paths[b] + [action_sample]
                renderer = RenderNet(self.x_beam_paths[b][epoch], self.y_beam_paths[b][epoch],
                                     x_samples[i], y_samples[i], imsize=self.imsize, fuzz=self.fuzz)
                sketch = renderer()
                sketch = torch.add(sketch, self.beam_sketches[b])
                sketch = (sketch - torch.min(sketch)) / (torch.max(sketch) - torch.min(sketch))
                sketches[i] = sketch

            losses = pixel_sketch_loss(gt_sketch, sketches)

            if b == 0:
                beam_losses = losses.data.numpy()[0]
                x_beam_samples = x_samples
                y_beam_samples = y_samples
                action_beam_samples = np.array([action_sample])
                all_sketches = sketches.clone()
            else:
                beam_losses = np.concatenate((beam_losses, losses.data.numpy()[0]))
                x_beam_samples = np.concatenate((x_beam_samples, x_samples))
                y_beam_samples = np.concatenate((y_beam_samples, y_samples))
                action_beam_samples = np.append(action_beam_samples, action_sample)
                all_sketches = torch.cat((all_sketches, sketches.clone()), dim=0)

        top_indexes = np.argsort(beam_losses)[:self.beam_width]
        _beam_sketches = Variable(torch.zeros((self.beam_width, 1, self.imsize, self.imsize)))

        for b in range(self.beam_width):
            # not each beam will update...
            beam_parent = int(np.floor(top_indexes[b] / self.n_samples))

            x_beam_path = self.x_beam_paths[beam_parent, :]
            y_beam_path = self.y_beam_paths[beam_parent, :]
            x_beam_path[epoch + 1] = x_beam_samples[top_indexes[b]]
            y_beam_path[epoch + 1] = y_beam_samples[top_indexes[b]]
            self.x_beam_paths[b, :] = x_beam_path
            self.y_beam_paths[b, :] = y_beam_path
            self.action_beam_paths[b].append(action_beam_samples[b])
            _beam_sketches[b] = all_sketches[top_indexes[b]]

        self.beam_sketches = _beam_sketches  # replace old with new
        self.x_beam_queue = np.array([x_beam_samples[top_indexes[b]]  # recreate queue
                                      for b in range(self.beam_width)])
        self.y_beam_queue = np.array([y_beam_samples[top_indexes[b]]
                                      for b in range(self.beam_width)])

        print('Train Epoch: {} \tLoss: {:.6f} \tParams: ({}, {})'.format(
              epoch, beam_losses[top_indexes[0]], x_beam_samples[top_indexes[0]],
              y_beam_samples[top_indexes[0]]))

        return all_sketches[top_indexes[0]]


if __name__ == "__main__":
    gt_sketch = gen_ground_truth()
    beamer = BeamSearch(5, 5, 11, beam_width=1, n_samples=10000,
                        n_iters=1, stdev=3, fuzz=0.1)
    sketch = beamer.train(gt_sketch, 0)
