from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import subprocess

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-device', type=int, default=0, help='0|1|2|3|4|5|6|7 [default: 0]')
    args = parser.parse_args()

    for hiddens_dim in [512, 256, 128, 64, 32, 16]:
        out_dir = '/mnt/visual_communication_dataset/trained_models_5_30_18/hiddens_fc6/%d' % hiddens_dim
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        command = 'CUDA_VISIBLE_DEVICES={device} python train_average.py fc6 --hiddens-dim {hiddens_dim} --train-test-split-dir ./train_test_split/1 --out-dir {out_dir} --cuda'.format(
            device=args.cuda_device, hiddens_dim=hiddens_dim, out_dir=out_dir)
        subprocess.call(command, shell=True)
