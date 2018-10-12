from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from train_average import load_checkpoint, AverageMeter
from dataset import ExhaustiveDataset, OBJECT_TO_CATEGORY


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str,
                        help='path to trained model file.')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='number of examples in a mini-batch [default: 10]')
    parser.add_argument('--train-test-split-dir', type=str, default='./train_test_split/1',
                        help='where to load train_test_split paths [default: ./train_test_split/1]')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.checkpoint_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model = model.cuda()

    test_dataset = ExhaustiveDataset(layer=model.layer, split='test',
                                     train_test_split_dir=args.train_test_split_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    object_acc_meter = 0
    category_acc_meter = 0
    pbar = tqdm(total=len(test_loader))

    for batch_idx, (sketch, sketch_object, sketch_context, sketch_path) in enumerate(test_loader):
        sketch = Variable(sketch, volatile=True)
        batch_size = len(sketch)
        sketch_object = sketch_object[0]
        sketch_category = OBJECT_TO_CATEGORY[sketch_object]

        if args.cuda:
            sketch = sketch.cuda()

        pred_logits = []
        photo_objects = []
        photo_categories = []
        photo_generator = test_dataset.gen_photos()
        for photo, photo_object, photo_path in photo_generator():
            photo_category = OBJECT_TO_CATEGORY[photo_object]
            photo = Variable(photo)
            if args.cuda:
                photo = photo.cuda()
            photo = (photo.repeat(batch_size, 1) if model.layer == 'fc6' else
                     photo.repeat(batch_size, 1, 1, 1))
            pred_logit = model(photo, sketch)
            pred_logits.append(pred_logit)
            photo_objects.append(photo_object)
            photo_categories.append(photo_category)

        pred_logits = torch.cat(pred_logits, dim=1)
        pred = F.softmax(pred_logits, dim=1)
        pred = torch.max(pred_logits, dim=1)[1]
        pred = pred.data[0]

        pred_sketch_object = photo_objects[pred]
        pred_sketch_category = photo_categories[pred]

        object_acc = pred_sketch_object == sketch_object
        category_acc = pred_sketch_category == sketch_category

        object_acc_meter += object_acc
        category_acc_meter += category_acc

        pbar.update()
    pbar.close()

    object_acc_meter /= float(len(test_loader))
    category_acc_meter /= float(len(test_loader))

    print('------------------------------')
    print(args.checkpoint_path)
    print('------------------------------')
    print('Object-level Accuracy: %f' % object_acc_meter)
    print('Category-level Accuracy: %f' % category_acc_meter)
    print('------------------------------')
    print('')

