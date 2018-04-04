from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cPickle
import numpy as np
import pandas as pd


if __name__ == "__main__":
    human_csv = '/mnt/visual_communication_dataset/sketchpad_basic_recog_group_data_augmented.csv'
    df = pd.read_csv(human_csv)
    sketchIDs = np.asarray(df['sketchID']).tolist()
    choices = np.asarray(df['choice']).tolist()
   
    object_csv = '/mnt/visual_communication_dataset/human_confusion_object_order.csv'
    object_order = np.asarray(pd.read_csv(object_csv))[:, 1].tolist()

    sketch_file = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_fc6/sketchpad_context_dict.pickle'
    with open(sketch_file) as fp:
        sketch_paths = cPickle.load(fp)
    sketch_paths = sketch_paths.keys()
    sketch_identifiers = []
    for sketch_path in sketch_paths:
        identifier = os.path.splitext(sketch_path)[0].split('_')[-3]
        trial = os.path.splitext(sketch_path)[0].split('_')[-1]
        identifier = identifier.split('-')[-1] + '_' + trial
        sketch_identifiers.append(identifier)
    sketch_id2path = dict(zip(sketch_identifiers, sketch_paths))

    choices = [object_order.index(choice) for choice in choices]
    sketchIDs = [sketch_id2path[sketchID] for sketchID in sketchIDs]
    result = np.array(zip(sketchIDs, choices)) 

    result_name = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_fc6/sketchpad_hard_labels.npy'
    np.save(result_name, result)
