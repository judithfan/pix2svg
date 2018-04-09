from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, csv
import subprocess
import torch
import numpy as np
import pandas as pd

def get_sketch_basenames(layer='fc6'):
    db_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_%s' % layer
    photos_dirname = os.path.join(db_path, 'photos')
    sketch_dirname = os.path.join(db_path, 'sketch')
    sketch_basepaths = os.listdir(sketch_dirname)
    valid_game_ids = pd.read_csv(os.path.join(db_path, 'valid_gameids_pilot2.csv'))['valid_gameids']
    valid_game_ids = np.asarray(valid_game_ids).tolist()
    # only keep sketches that are in the valid_gameids (some games are garbage)
    sketch_basepaths = [path for path in sketch_basepaths 
                        if os.path.basename(path).split('_')[1] in valid_game_ids]
    return sketch_basepaths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('num_targets', type=int, help='number of different "folds" to do')
    parser.add_argument('cuda_device', type=int, help='which CUDA device to use')
    parser.add_argument('out_superdir', type=str, help='where to save folders of files')
    parser.add_argument('model', type=str, help='ModelA|ModelB|ModelC|ModelD|ModelE|ModelF|ModelG|ModelH|ModelI|ModelJ|ModelK')
    # parser.add_argument('--soft-labels', action='store_true', default=False,
    #                     help='use soft or hard labels [default: False]')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='number of examples in a mini-batch [default: 64]')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate [default: 3e-4]')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs [default: 500]')
    parser.add_argument('--log-interval', type=int, default=10, help='how frequently to print stats [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    sketch_basepaths = get_sketch_basenames('fc6')
    test_choices = np.random.choice(np.arange(len(sketch_basepaths)), size=args.num_targets, replace=False)
    for i in xrange(args.num_targets):
        out_dir = os.path.join(args.out_superdir, 'fold-%d' % i)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        test_sketch_basepath = sketch_basepaths[test_choices[i]]
        cmd = 'CUDA_VISIBLE_DEVICES={device} python targeted.py {model} {test_sketch_basepath} --out-dir {out_dir} --batch-size {batch_size} --lr {lr} --epochs {epochs} --log-interval {log_interval} {cuda}'.format(
            device=args.cuda_device, model=args.model, test_sketch_basepath=test_sketch_basepath,
            batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, out_dir=out_dir, 
            log_interval=args.log_interval, cuda='--cuda' if args.cuda else '') 
        print('>>> %s' % cmd)
        subprocess.call(cmd, shell=True)

