"""As a control we should try raw distances between low-level 
features in VGG. We want to make the claim that our embeddings
are superior in simulating human behavior."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys

from glob import glob
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

from generator import list_files
from generator import preprocessing
from generator import alpha_composite_with_color


def extract_features(x, cnn, ix, classifier=False):
    if classifier:
        x = cnn.features(x)
        x = x.view(x.size(0), -1)
        classifier = list(cnn.classifier)[:ix + 1]
        for i in range(len(classifier)):
            x = classifier[i](x)
    else:
        features = list(cnn.features)[:ix + 1]
        for i in range(len(features)):
            x = features[i](x)
        x = x.view(x.size(0), -1)
    return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert string to embeddings')
    # sub-folder structure will be preserved. in other words, if imgfolder/ has
    # a/, b/, and c/, then outfolder/ will also have the same subfolders.
    parser.add_argument('imgfolder', type=str, help='path to where images are stored')
    parser.add_argument('outfolder', type=str, help='path to save text embeddings to')
    parser.add_argument('extension', type=str, help='jpg|png')
    parser.add_argument('layer_ix', type=int, help='which layer index to pull features from')
    parser.add_argument('--classifier', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--transparent', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    # check and create all sub-directories
    all_folders = [x[0] for x in os.walk(args.imgfolder)]
    for folder in all_folders:
        new_folder = folder.replace(args.imgfolder, args.outfolder)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

    # load the CNN
    cnn = models.vgg19(pretrained=True)
    cnn.eval()
    if args.cuda:
        cnn.cuda()

    image_paths = list_files(args.imgfolder, args.extension)
    n_images = len(image_paths)

    # store raw images in a batch so we can evaluate them using vgg
    image_jpg_batch, image_jpg_batches = [], []
    image_path_batch, image_path_batches = [], []

    for i in range(n_images):
        print('Loading image [{}/{}]'.format(i + 1, n_images))

        image_torch = Image.open(image_paths[i])
        if args.transparent:
            image_torch = alpha_composite_with_color(image_torch)
        image_torch = image_torch.convert('RGB')
        image_torch = preprocessing(image_torch).unsqueeze(0)
        image_jpg_batch.append(image_torch)
        image_path_batch.append(image_paths[i])

        if i % args.batch_size == 0:
            image_jpg_batch = torch.cat(image_jpg_batch, dim=0)
            image_jpg_batches.append(image_jpg_batch)
            image_path_batches.append(image_path_batch)
            image_jpg_batch = []
            image_path_batch = []

    if len(image_jpg_batch) > 0:
        image_jpg_batch = torch.cat(image_jpg_batch, dim=0)
        image_jpg_batches.append(image_jpg_batch)
        image_path_batches.append(image_path_batch)

    n_batches = len(image_jpg_batches)
    
    for i in range(n_batches):
        print('Getting embeddings [batch {}/{}]'.format(i + 1, n_batches))
        image_inputs = image_jpg_batches[i]
        image_inputs = Variable(image_inputs, volatile=True)

        if args.cuda:
            image_inputs = image_inputs.cuda()

        image_emb = extract_features(image_inputs, cnn, args.layer_ix, 
                                     classifier=args.classifier)
    
        image_emb = image_emb.cpu().data.numpy()
        batch_paths = image_path_batches[i]
        for j in range(len(image_inputs)):
            path_name = batch_paths[j].replace(args.imgfolder, args.outfolder)
            path_name = path_name.replace(args.extension, 'npy')
            np.save(path_name, image_emb[j])
