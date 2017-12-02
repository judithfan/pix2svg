"""Try to plot a manifold segmented by either category
or by instance level labels."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cPickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
from torch.autograd import Variable

from datasets import ContextFreePreloadedGenerator as Generator
from datasets import INSTANCE_IX2NAME_DICT, CATEGORY_IX2NAME_DICT
from model import load_checkpoint


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to where model is stored')
    parser.add_argument('--layer', type=str, help='conv_4_2|fc7', default='conv_4_2')
    parser.add_argument('--instance', action='store_true', 
                        help='if supplied, use instance level statistics instead of category statistics.')
    parser.add_argument('--pca_only', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.layer in ['conv_4_2', 'fc7']

    generator = Generator(train=False, batch_size=1, use_cuda=args.cuda, 
                          data_dir='/data/jefan/sketchpad_basic_fixedpose96_%s' % args.layer)
    examples = generator.make_generator()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()
    
    n_labels = 32 if args.instance else 4
    ix2name_dict = INSTANCE_IX2NAME_DICT if args.instance else CATEGORY_IX2NAME_DICT

    sketch_embeddings = torch.zeros((generator.size, 4, 1000))
    photo_embeddings = torch.zeros((generator.size, 4, 1000))
    sketch_labels = torch.zeros((generator.size, 4))
    photo_labels = torch.zeros((generator.size, 4))
    
    batch_idx = 0
    while True:
        try:
            photo, sketch, _, category, instance = examples.next()
            sketch = model.sketch_adaptor(sketch)
            photo = model.photo_adaptor(photo)

            # compute cosine similarity
            sketch = sketch - torch.mean(sketch, dim=1, keepdim=True)
            photo = photo - torch.mean(photo, dim=1, keepdim=True)

            sketch_embeddings[batch_idx, :, :] = sketch.cpu().data
            photo_embeddings[batch_idx, :, :] = photo.cpu().data
            if args.instance:
                sketch_labels[batch_idx, :] = instance.cpu().data
                photo_labels[batch_idx, :] = instance.cpu().data
            else:
                sketch_labels[batch_idx, :] = category.cpu().data
                photo_labels[batch_idx, :] = category.cpu().data

            batch_idx += 1
            print('Collecting examples [{}/{}].'.format(batch_idx, generator.size))
        except StopIteration:
            break

    sketch_embeddings = sketch_embeddings.view(-1, 1000).numpy()
    photo_embeddings = photo_embeddings.view(-1, 1000).numpy()
    sketch_labels = sketch_labels.view(-1).numpy()
    photo_labels = photo_labels.view(-1).numpy()
    n_sketch_embeddings = sketch_embeddings.shape[0]

    all_embeddings = np.concatenate((sketch_embeddings, photo_embeddings), axis=0)
    
    if not args.pca_only:
        pca_50 = PCA(n_components=50)
        tsne_2 = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        all_embeddings = pca_50.fit_transform(all_embeddings)
        all_embeddings = tsne_2.fit_transform(all_embeddings)
    else:
        pca_2 = PCA(n_components=2)
        all_embeddings = pca_2.fit_transform(all_embeddings)

    sketch_embeddings = all_embeddings[:n_sketch_embeddings, :]
    photo_embeddings = all_embeddings[n_sketch_embeddings:, :]

    # generate plot with all 8 embedding types in 1 thing.
    plt.figure()
    for i in xrange(n_labels):
        _sketch_embeddings = sketch_embeddings[sketch_labels == i]
        _photo_embeddings = photo_embeddings[photo_labels == i]
        plt.scatter(_sketch_embeddings[:, 0], _sketch_embeddings[:, 1],
                    alpha=0.1, edgecolors='none', label='%s-sketch' % ix2name_dict[i])
        plt.scatter(_photo_embeddings[:, 0], _photo_embeddings[:, 1],
                    alpha=0.1, edgecolors='none', label=ix2name_dict[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig('./manifold-{}.png'.format(
        'instance' if args.instance else 'category'))

    # generate plot with 2 subplots: 1 for sketches; 1 for photos
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    for i in xrange(n_labels):
        _sketch_embeddings = sketch_embeddings[sketch_labels == i]
        _photo_embeddings = photo_embeddings[photo_labels == i]
        ax1.scatter(_sketch_embeddings[:, 0], _sketch_embeddings[:, 1],
                    alpha=0.1, edgecolors='none', label=ix2name_dict[i])
        ax2.scatter(_photo_embeddings[:, 0], _photo_embeddings[:, 1],
                    alpha=0.1, edgecolors='none', label=ix2name_dict[i])

    ax1.legend()
    ax2.legend()
    ax1.set_title('Sketch Embeddings')
    ax2.set_title('Photo Embeddings')
    plt.tight_layout()
    plt.savefig('./manifold-{}-dual.png'.format(
        'instance' if args.instance else 'category'))

    # generate plot with either 4 or 32 subplots (1 for each little subclass)
    if args.instance:
        n_rows, n_cols = 4, 8
    else:
        n_rows, n_cols = 2, 2

    f, axarr = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    for i in xrange(n_labels):
        row_ix, col_ix = i // n_rows, i % n_cols
        _sketch_embeddings = sketch_embeddings[sketch_labels == i]
        _photo_embeddings = photo_embeddings[photo_labels == i]
        axarr[row_ix, col_ix].scatter(_sketch_embeddings[:, 0], _sketch_embeddings[:, 1],
                                      alpha=0.1, edgecolors='none', label='sketch')
        axarr[row_ix, col_ix].scatter(_photo_embeddings[:, 0], _photo_embeddings[:, 1],
                                      alpha=0.1, edgecolors='none', label='photo')
        axarr[row_ix, col_ix].legend()
        axarr[row_ix, col_ix].set_title(ix2name_dict[i])
    plt.tight_layout()
    plt.savefig('./manifold-{}-parts.png'.format(
        'instance' if args.instance else 'category'))
