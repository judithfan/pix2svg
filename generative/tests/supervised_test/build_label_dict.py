from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cPickle
import numpy as np
import pandas as pd


df         = pd.read_csv('/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_conv_4_2/sketchpad_basic_pilot2_group_data.csv')
game_ids   = np.asarray(df['gameID'])
trial_ids  = np.asarray(df['trialNum'])
target     = np.asarray(df['target'])
n_rows     = len(game_ids)

target_dict = {}
for i in xrange(n_rows):
    path = 'gameID_%s_trial_%d.npy' % (game_ids[i], trial_ids[i])
    target_dict[path] = target[i]

with open('/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_conv_4_2/sketchpad_label_dict.pickle', 'wb') as fp:
    cPickle.dump(target_dict, fp)

