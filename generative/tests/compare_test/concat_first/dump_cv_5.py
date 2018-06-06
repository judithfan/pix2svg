from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import subprocess

if __name__ == "__main__":
    for layer in ['fc6', 'conv42', 'pool1']:
        for i in xrange(5):
            print('Dumping files for (%s|%d)' % (layer, i))
            model_path = '/mnt/visual_communication_dataset/trained_models_6_5_18/%s/%d/model_best.pth.tar' % (layer, i + 1)
            out_dir = './dump_outputs_6_5_2018/%s/%d/' % (layer, i + 1)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            split_dir = './train_test_split/%d' % (i + 1)
            command = 'CUDA_VISIBLE_DEVICES=0 python dump.py {model} --train-test-split-dir {splitdir} --out-dir {outdir} --average-labels --overwrite-layer {layer} --cuda'.format(
                model=model_path, splitdir=split_dir, outdir=out_dir, layer=layer)
            subprocess.call(command, shell=True)
