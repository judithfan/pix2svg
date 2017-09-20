"""Training a joint embedding net will go much faster if we precompute
the embeddings for all images and sketches and save them to some shared 
directory as Numpy Arrays.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
sys.path.append('..')
from glob import glob

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


def cnn_predict(x, cnn):
    x = cnn.features(x)
    x = x.view(x.size(0), -1)

    classifier = list(cnn.classifier)[:4]
    for i in range(len(classifier)):
        x = classifier[i](x)
    
    return x


def list_files(path, ext='jpg'):
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert string to embeddings')
    # sub-folder structure will be preserved. in other words, if imgfolder/ has
    # a/, b/, and c/, then outfolder/ will also have the same subfolders.
    parser.add_argument('imgfolder', type=str, help='path to where images are stored')
    parser.add_argument('outfolder', type=str, help='path to save text embeddings to')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    # check and create all sub-directories
    all_folders = [x[0] for x in os.walk(args.outfolder)]
    for folder in all_folders:
        new_folder = folder.replace(args.imgfolder, args.outfolder)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

    # load the CNN
    cnn = models.vgg19(pretrained=True)
    cnn.eval()

    if args.cuda:
        cnn.cuda()

    for p in cnn.parameters():
        p.requires_grad = False

    preprocessing = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

    image_paths = list_files(args.imgfolder)
    n_images = len(image_paths)

    # store raw images in a batch so we can evaluate them using vgg
    image_jpg_batch, image_jpg_batches = [], []

    for i in range(n_images):
        print('Loading image [{}/{}]'.format(i + 1, n_images))

        image_torch = Image.open(image_paths[i])
        image_torch = image_torch.convert('RGB')
        image_torch = preprocessing(image_torch).unsqueeze(0)
        image_jpg_batch.append(image_torch)

        if i % args.batch_size == 0:
            image_jpg_batch = torch.cat(image_jpg_batch, dim=0)
            image_jpg_batches.append(image_jpg_batch)
            image_jpg_batch = []

    if len(image_jpg_batch) > 0:
        image_jpg_batch = torch.cat(image_jpg_batch, dim=0)
        image_jpg_batches.append(image_jpg_batch)

    n_batches = len(image_jpg_batches)
    
    image_emb_batches = []
    for i in range(n_batches):
        print('Getting embeddings [batch {}/{}]'.format(i + 1, n_batches))
        image_inputs = image_jpg_batches[i]
        image_inputs = Variable(image_inputs, volatile=True)

        if args.cuda:
            image_inputs = image_inputs.cuda()

        image_emb = cnn_predict(image_inputs, cnn)
        image_emb_batches.append(image_emb)

    image_embs = torch.cat(image_emb_batches, dim=0)
    image_embs = image_embs.cpu().data.numpy()
    assert(image_embs.shape[0] == len(image_ids))

    for i in range(n_images):
        print('Saving numpy object [{}/{}]'.format(i + 1, n_images))
        path_name = image_paths[i].replace(args.imgfolder, args.outfolder)
        np.save(path_name, image_embs[i])
