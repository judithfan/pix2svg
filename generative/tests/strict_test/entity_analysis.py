# Get mean/std performance per game -- should 2 numbers 
# per game (40 test games)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import random
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score

from convmodel import load_checkpoint
from convmodel import cosine_similarity
from referenceutils2 import EntityPreloadedGenerator


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to where model is stored')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # load this generator initially to get the test set...
    generator = EntityPreloadedGenerator(
        train=False, batch_size=25, use_cuda=args.cuda, closer_only=False,
        data_dir='/data/jefan/sketchpad_basic_fixedpose96_conv_4_2')
    _, render_paths = generator.train_test_split()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()
    
    dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    
    probas = defaultdict(lambda: [])
    labels = defaultdict(lambda: [])

    for i in xrange(generator.size):
        print('processing [{}/{}]'.format(i + 1, generator.size))
        render1_path = render_paths[i]
        sketch1_path = generator.target2sketch[render1_path]
        render2_path = random.choice(generator.target2distractors[render1_path])
        sketch2_path = generator.distractor2sketch[render2_path]
        
        assert sketch1_path.split('_')[1] == render1_path.split('_')[1]
        assert sketch2_path.split('_')[1] == render2_path.split('_')[1]
        assert render1_path.split('_')[1] == render2_path.split('_')[1]

        gameid = sketch1_path.split('_')[1]

        render1_dir = generator.path2folder[render1_path]
        sketch1_dir = generator.path2folder[sketch1_path]
        render2_dir = generator.path2folder[render2_path]
        sketch2_dir = generator.path2folder[sketch2_path]
        
        render1 = np.load(os.path.join(render1_dir, render1_path))[np.newaxis, ...]
        sketch1 = np.load(os.path.join(sketch1_dir, sketch1_path))[np.newaxis, ...]
        render2 = np.load(os.path.join(render2_dir, render2_path))[np.newaxis, ...]
        sketch2 = np.load(os.path.join(sketch2_dir, sketch2_path))[np.newaxis, ...]

        render = np.vstack((render1, render2, render1, render2))
        sketch = np.vstack((sketch1, sketch2, sketch2, sketch1))
        label = np.array((1, 1, 0, 0))

        render = Variable(torch.from_numpy(render)).type(dtype)
        sketch = Variable(torch.from_numpy(sketch)).type(dtype)
        label = Variable(torch.from_numpy(label), requires_grad=False).type(dtype)

        output = model(render, sketch)
        _output = output.cpu().data.numpy().flatten().tolist()
        _label = label.cpu().data.numpy().tolist()

        probas[gameid].extend(_output)
        labels[gameid].extend(_label)


    assert set(probas.keys()) == set(labels.keys())
    keys = probas.keys()
    auc_dict = dict()
    for key in keys:
        proba = np.array(probas[key])
        pred = np.rint(proba).astype(int)
        label = np.array(labels[key]).astype(int)

        auc = accuracy_score(label, pred)
        auc_dict[key] = auc
