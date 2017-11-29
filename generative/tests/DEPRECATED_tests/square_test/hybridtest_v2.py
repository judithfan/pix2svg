from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys; sys.path.append('../..')
import numpy as np

import torch
import torch.optim as optim

from beamsearch import PixelBeamSearch
from linerender import SketchRenderNet
from rendertest import gen_ground_truth


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gt_sketch = gen_ground_truth()
    beamer = PixelBeamSearch(2, 2, 11, beam_width=1, n_samples=10000,
                             n_iters=4, stdev=3, fuzz=0.1)
    for i in range(4):
        sketch = beamer.train(i, gt_sketch)

    x_paths, y_paths = beamer.gen_paths()
    print(zip(x_paths, y_paths))
    plt.matshow(sketch[0].data.numpy())
    plt.savefig('./sketch_beam.png')

    renderer = SketchRenderNet(x_paths, y_paths, imsize=11, fuzz=1.0)
    optimizer = optim.Adam(renderer.parameters(), lr=5e-4)

    def train(renderer, optimizer, epoch):
        renderer.train()
        optimizer.zero_grad()
        sketch = renderer()
        loss = torch.sum(torch.pow(gt_sketch - sketch, 2))  # l2 loss
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.data[0]))

    for i in range(500):
        train(renderer, optimizer, i)

    x_paths = renderer.x_list.data.numpy().tolist()
    y_paths = renderer.y_list.data.numpy().tolist()
    renderer = SketchRenderNet(x_paths, y_paths, imsize=11, fuzz=0.1)
    sketch = renderer()
    print(zip(x_paths, y_paths))
    plt.matshow(sketch[0][0].data.numpy())
    plt.savefig('./sketch_tune.png')
