from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms

from distribtest import list_files, load_image


class RetrieverNet(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(4096, 1250)  # 1,250 classes

    def forward(self, x):
        x = self.fc(x)
        return F.softmax(x)


def gen_class_order():
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'
    sketch_paths = list_files(sketch_dir, ext='png')

    classes = []
    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1] 
        classes.append(sketch_folder)

    return classes


def gen_airplane_order(path):
    airplane_dir = '/home/jefan/full_sketchy_dataset/photos/airplane'
    return [i for i in os.listdir(airplane_dir) if '.jpg' in i]


def class_test_generator(imsize=256, use_cuda=False):
    # todo -- split into train/validation/test set
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'
    sketch_paths = list_files(sketch_dir, ext='png')
    random.shuffle(sketch_paths)

    class_order = gen_class_order()
    n_classes = len(class_order)

    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
        
        label_i = class_order.index(sketch_folder)
        label = Variable(torch.zeros(n_classes))
        label[label_i] = 1
        if use_cuda:
            label = label.cuda()

        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)
        yield (sketch, label)


def airplane_test_generator(imsize=256, use_cuda=False):
    # todo -- split into train/validation/test set
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches/airplane'
    sketch_paths = list_files(sketch_dir, ext='png')
    random.shuffle(sketch_paths)

    airplane_order = gen_airplane_order()
    n_airplanes = len(airplane_order)

    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        photo_filename = sketch_filename.split('-')[0] + '.jpg'

        label_i = airplane_order.index(photo_filename)
        label = Variable(torch.zeros(n_airplanes))
        label[label_i] = 1
        if use_cuda:
            label = label.cuda()

        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)
        yield (sketch, label)


def deactivate(net):
    net.eval()
    for p in net.parameters():
        p.requires_grad = False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
 
    assert args.experiment in ['class', 'airplane']

    use_cuda = args.cuda and torch.cuda.is_available()

    vgg19 = models.vgg19(pretrained=True)
    vgg19_features = deepcopy(vgg19.features)
    vgg19_classifier = deepcopy(vgg19.classifier)

    # remove last layer of classifier
    vgg19_classifier = nn.Sequential(*(list(vgg19_classifier.children())[:-1]))

    deactivate(vgg19_features)
    deactivate(vgg19_classifier)

    retriever = RetrieverNet()

    if use_cuda:
        vgg19_features.cuda()
        vgg19_classifier.cuda()
        retriever.cuda()

    if args.experiment == 'class':
        generator = class_test_generator(imsize=224, use_cuda=use_cuda)
    else:
        generator = airplane_test_generator(imsize=224, use_cuda=use_cuda)
    optimizer = optim.Adam(retriever.parameters(), lr=args.lr)


    def train(epoch):
        retriever.train()
        b = 0  # number of batches
        n = 0  # number of examples
        quit = False 

        if generator:
            while True:
                sketch_batch = Variable(torch.zeros(args.batch, 3, 224, 224))
                label_batch = Variable(torch.zeros(args.batch))

                if use_cuda:
                    sketch_batch = sketch_batch.cuda()

                for b in range(args.batch):
                    try:
                        sketch, label = generator.next()
                        sketch_batch[b] = sketch
                        label_batch[b] = label
                    except StopIteration:
                        quit = True
                        break

                sketch_batch = sketch_batch[:b + 1]
                label_batch = label_batch[:b + 1]  

                sketch_batch = vgg19_features(sketch_batch) 
                sketch_batch = vgg19_classifier(sketch_batch)

                optimizer.zero_grad()
                output = retriever(sketch_batch)

                loss = F.nll_loss(output, label_batch)
                loss.backward()
                optimizer.step()

                n += (b + 1)

                if b % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, b * sketch_batch.size()[0], '--', loss.data[0]))

                if quit:
                    break


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # todo: add save checkpointing

