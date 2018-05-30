from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import numpy as np
from dataset import VisualDataset

if __name__ == "__main__":
    # define a list of random seeds
    # this will generate the cross-validation we seek
    random_seeds = np.random.randint(0, 1000, size=5)
    random_seeds[0] = 42
    for i in xrange(5):
        dset = VisualDataset(layer='fc6', split='train', average_labels=False, 
                             photo_transform=None, sketch_transform=None,
                             train_test_split_dir='./train_test_split/%d/' % (i + 1),
                             random_seed=random_seeds[i])

