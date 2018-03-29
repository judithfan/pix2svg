from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import cPickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.autograd import Variable

from model import SketchNet
from dataset import SketchPlus32Photos
from train_category import load_checkpoint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./trained_models/fc6/model_best.pth.tar',
                        help='where trained models are stored [default: ./trained_models/fc6/model_best.pth.tar]')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # load the trained model
    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda() 

    # this is the order of the RDM.
    object_order = pd.read_csv('/mnt/visual_communication_dataset/human_confusion_object_order.csv')
    object_order = np.asarray(object_order['object_name']).tolist() 

    # this is to tell us about closer or further
    context_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_fc6/sketchpad_context_dict.pickle'
    with open(context_path) as fp:
        context_dict = cPickle.load(fp)   

    label_path = '/mnt/visual_communication_dataset/sketchpad_basic_fixedpose96_fc6/sketchpad_label_dict.pickle'
    with open(label_path) as fp:
        label_dict = cPickle.load(fp)

    close_sketch_features = defaultdict(lambda: [])
    far_sketch_features = defaultdict(lambda: [])

    # load the dataset
    loader = torch.utils.data.DataLoader(SketchPlus32Photos(layer=model.layer, return_paths=True),
                                         batch_size=1, shuffle=False)
    # loop through dataset, compute embeddings and store them.
    pbar = tqdm(total=len(loader))
    for batch_idx, (photos, sketch, photo_paths, sketch_path, label) in enumerate(loader):
        sketch = Variable(sketch, volatile=True)
        if args.cuda:
            sketch = sketch.cuda()
        sketch = model.sketch_adaptor(sketch).cpu().data.numpy()
        sketch_context = context_dict[os.path.basename(sketch_path[0])]
        sketch_object = label_dict[os.path.basename(sketch_path[0])]
 
        if sketch_context == 'closer':
            close_sketch_features[sketch_object].append(sketch[0])
        elif sketch_context == 'further':
            far_sketch_features[sketch_object].append(sketch[0])
        else:
            raise Exception('Context %s is not recognized' % sketch_context)
        
        if batch_idx == 0:  # we only have to do this once. 
            photos = Variable(photos, volatile=True)
            if args.cuda:
                photos = photos.cuda()
            photos = torch.cat([model.photo_adaptor(photos[:, i]).unsqueeze(1).cpu().data
                                for i in xrange(32)], dim=1)
            render_features = photos.numpy()[0]

        pbar.update()
    pbar.close()

    close_sketch_features = np.array([np.mean(np.array(close_sketch_features[object_name]), axis=0) 
                                      for object_name in object_order])
    far_sketch_features = np.array([np.mean(np.array(far_sketch_features[object_name]), axis=0)
                                    for object_name in object_order])
    
    close_rdm = np.corrcoef(np.vstack((render_features, close_sketch_features)))
    close_rdm = close_rdm[:32, 32:]
    far_rdm = np.corrcoef(np.vstack((render_features, far_sketch_features)))
    far_rdm = far_rdm[:32, 32:]
    diff_rdm = close_rdm - far_rdm 
    
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import seaborn as sns
    sns.set_style('whitegrid')
    
    plt.figure()
    ax = sns.heatmap(close_rdm, cmap="YlGnBu")
    plt.savefig('./close_rdm.pdf')

    plt.figure()
    ax = sns.heatmap(far_rdm, cmap="YlGnBu")
    plt.savefig('./far_rdm.pdf')

    plt.figure()
    ax = sns.heatmap(diff_rdm, cmap="YlGnBu")
    plt.savefig('./diff_rdm.pdf')    

