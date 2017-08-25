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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')

    gt_sketch = gen_ground_truth()

    beam_width = 2
    n_samples = 100
    n_iters = 10
    stdev = 2

    x_beam_queue = np.ones(beam_width) * 5
    y_beam_queue = np.ones(beam_width) * 5

    x_beam_paths = np.zeros((beam_width, n_iters + 1))
    y_beam_paths = np.zeros((beam_width, n_iters + 1))
    action_beam_paths = [[] for b in range(beam_width)]
    x_beam_paths[:, 0] = 5
    y_beam_paths[:, 0] = 5

    beam_sketches = Variable(torch.zeros((beam_width, 1, 11, 11)))

    def train(epoch):
        for b in range(beam_width):
            samples = sample_endpoint(x_beam_queue[b], y_beam_queue[b],
                                      std=stdev, size=n_samples,
                                      min_x=0, max_x=11, min_y=0, max_y=11)

            x_samples, y_samples = samples[:, 0], samples[:, 1]
            action_sample = sample_action()

            sketches = Variable(torch.zeros((n_samples, 1, 11, 11)))

            for i in range(n_samples):
                action_path = action_beam_paths[b] + [action_sample]
                renderer = RenderNet(x_beam_paths[b][-1], y_beam_paths[b][-1],
                                     x_samples[i], y_samples[i], imsize=11)
                sketch = renderer()
                sketch = torch.add(sketch, beam_sketches[b])
                # it's possible combining with earlier sketches goes over 1.
                # instead of normalizing, we should just clamp at 1.
                sketch = sketch.clamp(0, 1)
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

        top_indexes = np.argsort(beam_losses)[::-1][:beam_width]
        _beam_sketches = Variable(torch.zeros((beam_width, 1, 11, 11)))

        for b in range(beam_width):
            beam_parent = int(np.floor(top_indexes[b] / n_samples))

            x_beam_path = x_beam_paths[beam_parent, :]
            y_beam_path = y_beam_paths[beam_parent, :]
            x_beam_path[epoch + 1] = x_beam_samples[top_indexes[b]]
            y_beam_path[epoch + 1] = y_beam_samples[top_indexes[b]]
            x_beam_paths[b, :] = x_beam_path
            y_beam_paths[b, :] = y_beam_path
            action_beam_paths[b].append(action_beam_samples[b])

            beam_sketch = beam_sketches[beam_parent]
            beam_sketch = torch.add(beam_sketch, all_sketches[top_indexes[b]])
            beam_sketch = beam_sketch.clamp(0, 1)
            _beam_sketches[b] = beam_sketch

        beam_sketches = _beam_sketches  # replace old with new
        x_beam_queue = np.array([x_beam_samples[top_indexes[b]]  # recreate queue
                                for b in range(beam_width)])
        y_beam_queue = np.array([y_beam_samples[top_indexes[b]]
                                for b in range(beam_width)])

        print('Train Epoch: {} \tLoss: {:.6f}'.format(
              epoch, beam_losses[top_indexes[0]]))

        return all_sketches[top_indexes[0]]


    for i in range(n_iters):
        sketch = train(i)

    plt.matshow(sketch[0][0].data.numpy())
    plt.show()
