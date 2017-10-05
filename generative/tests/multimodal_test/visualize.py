"""Given a distance.npy generated from apply.py, we can visualize 
each distribution appropriately.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from generators import (SAME_PHOTO_EX, SAME_CLASS_EX, 
                        DIFF_CLASS_EX, NOISE_EX)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('distance_path', type=str, help='where to find saved distances.')
    parser.add_argument('plot_path', type=str, help='where to save plot')
    args = parser.parse_args()

    distances = np.load(args.distance_path)
    dists = distances[:, 0]
    types = distances[:, 1]
    
    plt.figure(figsize=(12, 6))
    plt.hist(dists[types == SAME_PHOTO_EX], 50, facecolor='r', alpha=0.3, 
             label='D(photo,sketch_same_photo)')
    plt.hist(dists[types == SAME_CLASS_EX], 50, facecolor='y', alpha=0.3, 
             label='D(photo, sketch_same_class)')
    plt.hist(dists[types == DIFF_CLASS_EX], 50, facecolor='b', alpha=0.3, 
             label='D(photo, sketch_diff_class)')
    plt.hist(dists[types == NOISE_EX], 50, facecolor='g', alpha=0.3, 
             label='D(photo, sketch_noise)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=14)
    plt.xlabel('Correlation', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig(args.plot_path)
