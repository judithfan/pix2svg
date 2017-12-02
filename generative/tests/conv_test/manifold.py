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

    sketch_embeddings = torch.zeros((generator.size, 1000))
    photo_embeddings = torch.zeros((generator.size, 1000))
    sketch_labels = []
    photo_labels = []
    
    batch_idx = 0
    while True:
        try:
            photo, sketch, _, category, instance = examples.next()
            sketch = model.sketch_adaptor(sketch)
            photo = model.photo_adaptor(photo)

            # compute cosine similarity
            sketch = sketch - torch.mean(sketch, dim=1, keepdim=True)
            photo = photo - torch.mean(photo, dim=1, keepdim=True)

            sketch_embeddings[batch_idx, :] = sketch.cpu().data[0]
            sketch_embeddings[batch_idx, :] = photo.cpu().data[0]

            if args.instance:
                sketch_labels.append(instance[0])
                photo_labels.append(instance[0])
            else:
                sketch_labels.append(category[0])
                photo_labels.append(category[0])

            batch_idx += 1
            print('Collecting examples [{}/{}].'.format(batch_idx, generator.size))
        except StopIteration:
            break

    sketch_embeddings = sketch_embeddings.numpy()
    photo_embeddings = photo_embeddings.numpy()
    all_embeddings = np.concatenate((sketch_embeddings, photo_embeddings), axis=0)

    sketch_labels = np.array(sketch_labels)
    photo_labels = np.array(photo_labels)

    pca_50 = PCA(n_components=50)
    tsne_2 = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

    all_embeddings = pca_50.fit_transform(all_embeddings)
    all_embeddings = tsne_2.fit_transform(all_embeddings)

    sketch_embeddings = all_embeddings[:generator.size, :]
    photo_embeddings = all_embeddings[generator.size:, :]

    plt.figure()
    for i in xrange(n_labels):
        _sketch_embeddings = sketch_embeddings[sketch_labels == i]
        _photo_embeddings = photo_embeddings[photo_labels == i]
        plt.scatter(_sketch_embeddings[:, 0], _sketch_embeddings[:, 1],
                    alpha=0.1, edgecolors='none', label='%s-sketch' % ix2name_dict[i])
        plt.scatter(_photo_embeddings[:, 0], _photo_embeddings[:, 1],
                    alpha=0.1, edgecolors='none', label=ix2name_dict[i])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('./manifold-{}.png'.format(
        'instance' if args.instance else 'category'))
