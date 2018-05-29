from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import subprocess

if __name__ == "__main__":
    for layer in ['fc6', 'conv42', 'pool1']:
        for i in xrange(5):
            print('Dumping files for ')
            model_path = './trained_models/%s/%d/model_best.pth.tar' % (layer, i + 1)
            out_dir = './trained_models/%s/%d/' % (layer, i + 1)
            command = 'CUDA_VISIBLE_DEVICES=0 python dumpy.py {model} --out-dir {outdir} --average-labels --cuda'.format(
                model=model_path, outdir=out_dir)
            subprocess.call(command, shell=True)
