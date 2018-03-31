from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import cPickle
import numpy as np
import pandas as pd


if __name__ == "__main__":
    targets_dir = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96/target'
    order_path = '/mnt/visual_communication_dataset/human_confusion_object_order.csv'
    out_dir = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96/photos'

    order = np.asarray(pd.read_csv(order_path))[:, 1]
    targets_localpath = os.listdir(targets_dir)
 
    paths_dict = {}

    for path in targets_localpath:
        targets_path = os.path.join(targets_dir, path)
        object_type = os.path.splitext(path)[0].split('_')[-1]
        paths_dict[object_type] = targets_path
   
    for object_type in order:
        path = paths_dict[object_type]
        shutil.copyfile(path, os.path.join(out_dir, '%s.png' % object_type))    

