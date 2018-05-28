from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import subprocess

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('layer', type=str, help='fc6|conv42|pool1')
    parser.add_argument('--cuda-device', type=int, default=0, help='0|1|2|3 [default: 0]')
    args = parser.parse_args()

    for i in xrange(5):
        out_dir = './trained_models/%s/%d' % (args.layer, i + 1)
        train_test_split_dir = './train_test_split/%d' % (i + 1)
        command = 'CUDA_VISIBLE_DEVICES={device} python train_average.py {layer} --train-test-split-dir {split_dir} --out-dir {out_dir} --cuda'.format(
            device=args.cuda_device, layer=args.layer, split_dir=train_test_split_dir, out_dir=out_dir)
        subprocess.call(command, shell=True)
