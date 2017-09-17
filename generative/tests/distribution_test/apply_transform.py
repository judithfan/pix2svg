"""Similar to transformtest.py but takes a trained model and 
applies it to the test dataset.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import os

import torch
from torch.autograd import Variable
import torchvision.models as models

from transformtest import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model')
    args.parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    _, generator = reset_generators()
    _, n_data = train_test_size() 

    cnn = models.vgg19()
    cnn.eval()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()

    if args.cuda:
        cnn.cuda()
        model.cuda()


    def test(epoch):
        losses = AverageMeter()
        constraints = AverageMeter()
        model.eval()
        quit = False

        while True:
            photo = Variable(torch.zeros(args.batch_size, 3, 224, 224))
            sketch = Variable(torch.zeros(args.batch_size, 3, 224, 224))
  
            if args.cuda:
                photo, sketch = photo.cuda(), sketch.cuda() 

            for b in range(args.batch_size):
                try:
                    _photo, _sketch = test_generator.next()
                    photo[b] = _photo
                    sketch[b] = _sketch
                except StopIteration:
                    quit = True
                    break

            photo, sketch = photo[:b + 1], sketch[:b + 1]
            photo_emb, sketch_emb = cnn_predict(photo, cnn), cnn_predict(sketch, cnn)
            photo_emb = model(photo_emb)
            
            loss = torch.norm(photo_emb - sketch_emb, p=2)
            losses.update(loss.data[0], b)

            if quit: 
                break

        print('Test Epoch: {}\tAverage Distance: {:.6f}\tAverage Constraint: {:.6f}'.format(
            epoch, losses.avg, constraints.avg))
        
        return losses.avg


    test(0)
