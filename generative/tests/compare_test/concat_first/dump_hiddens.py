from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import subprocess

if __name__ == "__main__":
    for hiddens_dim in [512, 256, 128, 64, 32, 16]:    
        print('Dumping files for (%d)' % hiddens_dim)
        model_path = '/mnt/visual_communication_dataset/trained_models_5_30_18/hiddens_fc6/%d/model_best.pth.tar' % hiddens_dim
        out_dir = './dump_hiddens_outputs/%d/' % hiddens_dim
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        command = 'CUDA_VISIBLE_DEVICES=7 python dump.py {model} --train-test-split-dir ./train_test_split/1 --out-dir {outdir} --average-labels --overwrite-layer fc6 --cuda'.format(model=model_path, outdir=out_dir)
        subprocess.call(command, shell=True)

