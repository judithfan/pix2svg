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

from beamsearch import PixelBeamSearch
from rendertest import gen_ground_truth


if __name__ == "__main__":
    gt_sketch = gen_ground_truth()
    beamer = PixelBeamSearch(5, 5, 11, beam_width=1, n_samples=10000,
                             n_iters=1, stdev=3, fuzz=0.1)
    sketch = beamer.train(0, gt_sketch)
