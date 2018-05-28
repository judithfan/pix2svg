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
    for i in xrange(1, 5):
        dset = VisualDataset(layer='fc6', split='train', average_labels=False, 
                             photo_transform=None, sketch_transform=None, 
                             random_seed=random_seeds[i])
        shutil.move('./train_split.json', './train_test_split/%d/train_split.json' % (i + 1))
        shutil.move('./val_split.json', './val_test_split/%d/val_split.json' % (i + 1))
        shutil.move('./test_split.json', './test_test_split/%d/test_split.json' % (i + 1))
