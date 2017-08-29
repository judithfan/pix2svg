from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys; sys.path.append('../..')
import numpy as np

from beamsearch import PixelBeamSearch
from rendertest import gen_ground_truth


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gt_sketch = gen_ground_truth()
    beamer = PixelBeamSearch(2, 2, 11, beam_width=1, n_samples=100,
                             n_iters=1, stdev=4, fuzz=0.1)
    for i in range(4):
        sketch = beamer.train(i, gt_sketch)

    x_paths, y_paths = beamer.gen_paths()
    plt.matshow(sketch[0].data.numpy())
    plt.savefig('./sketch_beam.png')

    renderer = SketchRenderNet(x_paths, y_paths, imsize=11, fuzz=1.0)
    optimizer = optim.SGD(renderer.parameters(), lr=1e-2, momentum=0.5)

    def train(renderer, optimizer, epoch):
        renderer.train()
        optimizer.zero_grad()
        sketch = renderer()
        loss = torch.sum(torch.pow(gt_sketch - sketch, 2))  # l2 loss
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.data[0])

    for i in range(250):
        train(renderer, optimizer, i)

    sketch = renderer()
    plt.matshow(sketch[0].data.numpy())
    plt.savefig('./sketch_tune.png')
