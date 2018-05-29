from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import subprocess

if __name__ == "__main__":
    for layer in ['fc6', 'conv42', 'pool1']:
        for i in xrange(5):
            print('Dumping files for (%s|%d)' % (layer, i))
            model_path = './trained_models/%s/%d/model_best.pth.tar' % (layer, i + 1)
            out_dir = './trained_models/%s/%d/' % (layer, i + 1)
            command = 'CUDA_VISIBLE_DEVICES=0 python dump.py {model} --out-dir {outdir} --average-labels --overwrite-layer {layer} --cuda'.format(
                model=model_path, outdir=out_dir, layer=layer)
            subprocess.call(command, shell=True)
