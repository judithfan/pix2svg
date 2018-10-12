r"""Loop through the 5 splits and run `get_similarity.py` for each model."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import subprocess


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('models_dir', type=str,
                        help='root where trained models are stored.')
    parser.add_argument('--cuda-device', type=int, default=0, help='0|1|2|3 [default: 0]')
    args = parser.parse_args()

    for layer in ['conv42', 'pool1', 'fc6']:
        for i in xrange(5):
            print('Testing model for (%s|%d)' % (layer, i))

            model_path = '%s/%s/%d/model_best.pth.tar' % (args.models_dir, layer, i + 1)
            split_dir = os.path.join('./train_test_split', '%d' % (i + 1))
            command = 'CUDA_VISIBLE_DEVICES={device} python test_average.py {model} --train-test-split-dir {splitdir} --cuda'.format(
                device=args.cuda_device, model=model_path, splitdir=split_dir)

            # call bash to run everything
            subprocess.call(command, shell=True)
