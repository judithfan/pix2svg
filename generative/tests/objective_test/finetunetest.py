from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms

from distribtest import list_files, load_image


class RetrieverNet(nn.Module):
    def __init__(self, n_class):
        super(RetrieverNet, self).__init__()
        self.fc = nn.Linear(4096, n_class)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x)


def gen_class_order():
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'
    sketch_paths = list_files(sketch_dir, ext='png')

    classes = []
    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1] 
        classes.append(sketch_folder)
    
    classes = np.array(classes)
    classes = np.unique(classes)

    return classes.tolist()


def gen_airplane_order():
    airplane_dir = '/home/jefan/full_sketchy_dataset/photos/airplane'
    return [i for i in os.listdir(airplane_dir) if '.jpg' in i]


def class_generator(imsize=256, use_cuda=False, train=True):
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'
    sketch_paths = list_files(sketch_dir, ext='png')
    # for training, we will use sketches 1 --> 5
    if train:
        sketch_paths = [i for i in sketch_paths if int(i.split('.')[0].split('-')[-1]) <= 5]
    else:  # test <-- this uses sketches after 5 
        sketch_paths = [i for i in sketch_paths if int(i.split('.')[0].split('-')[-1]) > 5]
    random.shuffle(sketch_paths)

    class_order = gen_class_order()
    n_classes = len(class_order)

    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
        
        label_i = class_order.index(sketch_folder)
        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)
        yield (sketch, label_i, sketch_path)


def class_generator_size(train=True):
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'
    sketch_paths = list_files(sketch_dir, ext='png')
    # for training, we will use sketches 1 --> 5
    if train:
        sketch_paths = [i for i in sketch_paths if int(i.split('.')[0].split('-')[-1]) <= 5]
    else:  # test <-- this uses sketches after 5
        sketch_paths = [i for i in sketch_paths if int(i.split('.')[0].split('-')[-1]) > 5]
    return len(sketch_paths)


def airplane_generator(imsize=256, use_cuda=False, train=True):
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches/airplane'
    sketch_paths = list_files(sketch_dir, ext='png')
    # for training, we will use sketches 1 --> 5
    if train:
        sketch_paths = [i for i in sketch_paths if int(i.split('.')[0].split('-')[-1]) <= 5]
    else:  # test <-- this uses sketches after 5 
        sketch_paths = [i for i in sketch_paths if int(i.split('.')[0].split('-')[-1]) > 5]
    random.shuffle(sketch_paths)

    airplane_order = gen_airplane_order()
    n_airplanes = len(airplane_order)

    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        photo_filename = sketch_filename.split('-')[0] + '.jpg'

        label_i = airplane_order.index(photo_filename)
        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)
        yield (sketch, label)


def airplane_generator_size(train=True):
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches/airplane'	    
    sketch_paths = list_files(sketch_dir, ext='png')
    # for training, we will use sketches 1 --> 5
    if train:
        sketch_paths = [i for i in sketch_paths if int(i.split('.')[0].split('-')[-1]) <= 5]
    else:  # test <-- this uses sketches after 5
        sketch_paths = [i for i in sketch_paths if int(i.split('.')[0].split('-')[-1]) > 5]
    return len(sketch_paths)


def deactivate(net):
    net.eval()
    for p in net.parameters():
        p.requires_grad = False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str, help='class|airplane')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
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

    if args.experiment == 'class':
        train_generator = class_generator(imsize=224, use_cuda=use_cuda, train=True)
        test_generator = class_generator(imsize=224, use_cuda=use_cuda, train=False)
        n_outputs = 125
        train_generator_size = class_generator_size(train=True)
	test_generator_size = class_generator_size(train=False)
    else:
        train_generator = airplane_generator(imsize=224, use_cuda=use_cuda, train=True)
        test_generator = airplane_generator(imsize=224, use_cuda=use_cuda, train=False)
        n_outputs = 100
	train_generator_size = airplane_generator_size(train=True)
	test_generator_size = airplane_generator_size(train=False)

    retriever = RetrieverNet(n_outputs)
    optimizer = optim.Adam(retriever.parameters(), lr=args.lr)
    
    if use_cuda:
        vgg19_features.cuda()
        vgg19_classifier.cuda()
        retriever.cuda()


    def train(epoch):
        retriever.train()
        b = 0  # number of batches
        n = 0  # number of examples
        quit = False 

        if train_generator:
            while True:
                sketch_batch = Variable(torch.zeros(args.batch, 3, 224, 224))
                label_batch = Variable(torch.zeros(args.batch))

                if use_cuda:
                    sketch_batch = sketch_batch.cuda()
                    label_batch = label_batch.cuda()

                for b in range(args.batch):
                    try:
                        sketch, label, _ = train_generator.next()
                        sketch_batch[b] = sketch[0]
                        label_batch[b] = label
                    except StopIteration:
                        quit = True
                        break

                sketch_batch = sketch_batch[:b + 1]
                label_batch = label_batch[:b + 1]  

                sketch_batch = vgg19_features(sketch_batch) 
                sketch_batch = sketch_batch.view(args.batch, -1)
                sketch_batch = vgg19_classifier(sketch_batch)
                label_batch = label_batch.long()
    
                optimizer.zero_grad()
                output = retriever(sketch_batch)

                loss = F.nll_loss(output, label_batch)
                loss.backward()
                optimizer.step()

                n += (b + 1)

                if b % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, n, train_generator_size, loss.data[0]))

                if quit:
                    break


    def test(epoch):
        retriever.eval()
        b = 0  # number of batches
        n = 0  # number of examples
        quit = False 

        acc_meter = AverageMeter()

        if test_generator:
            while True:
                sketch_batch = Variable(torch.zeros(args.batch, 3, 224, 224))
                label_batch = Variable(torch.zeros(args.batch))

                if use_cuda:
                    sketch_batch = sketch_batch.cuda()

                for b in range(args.batch):
                    try:
                        sketch, label, _ = test_generator.next()
                        sketch_batch[b] = sketch[0]
                        label_batch[b] = label
                    except StopIteration:
                        quit = True
                        break

                sketch_batch = sketch_batch[:b + 1]
                label_batch = label_batch[:b + 1]  

                sketch_batch = vgg19_features(sketch_batch) 
                sketch_batch = sketch_batch.view(args.batch, -1)
                sketch_batch = vgg19_classifier(sketch_batch)

                output_batch = retriever(sketch_batch)

                output_batch_np = output_batch.cpu().data.numpy()
                label_batch_np = label_batch.cpu().data.numpy()
                output_batch_ix = np.argmax(output_batch_np, dim=1)
                label_batch_ix = np.argmax(label_batch_np)

                acc = accuracy_score(label_batch_ix, output_batch_ix)
                acc_meter.update(acc, args.batch)
                n += (b + 1)

                if b % args.log_interval == 0:
                    print('Test Epoch: {} [{}/{}]\tAcc: {:.6f}'.format(
                        epoch, n, test_generator_size, acc_meter.avg))

                if quit:
                    break

            print('Average Total Accuracy: {}'.format(acc_meter.avg))


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
