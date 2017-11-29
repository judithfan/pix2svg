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
    beamer = PixelBeamSearch(5, 5, 11, beam_width=1, n_samples=10000,
                             n_iters=1, stdev=3, fuzz=0.1)
    sketch = beamer.train(0, gt_sketch)

    plt.matshow(sketch[0].data.numpy())
    plt.savefig('./sketch.png')
