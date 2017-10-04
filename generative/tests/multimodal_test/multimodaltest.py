"""A linear/nonlinear transformation from sketch to photo may 
not be enough -- the one-way projection may be too difficult.
We can instead project both the sketch and the photo into a 
shared embeddding space. Then we can randomly sample negatives 
from the same class and from the different class.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import random
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.autograd import Variable

import torchvision.models as models
from sklearn.metrics import accuracy_score
from precompute_vgg import list_files

import sys
sys.path.append('../distribution_test')
from distribtest import cosine_similarity

EMBED_NET_TYPE = 0
CONV_EMBED_NET_TYPE = 1


class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
        self.photo_adaptor = AdaptorNet(4096, 2048, 1000)
        self.sketch_adaptor = AdaptorNet(4096, 2048, 1000)
        self.fusenet = FuseClassifier()

    def forward(self, photo_emb, sketch_emb):
        photo_emb = self.photo_adaptor(photo_emb)
        sketch_emb = self.sketch_adaptor(sketch_emb)
        return self.fusenet(photo_emb, sketch_emb)


class AdaptorNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(AdaptorNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.drop1(x)
        x = F.elu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


class ConvEmbedNet(nn.Module):
    def __init__(self):
        super(ConvEmbedNet, self).__init__()
        self.photo_adaptor = ConvAdaptorNet(64, 8, 64, 64, 2048, 1000)
        self.sketch_adaptor = ConvAdaptorNet(64, 8, 64, 64, 2048, 1000)
        self.fusenet = FuseClassifier()

    def forward(self, photo_emb, sketch_emb):
        photo_emb = self.photo_adaptor(photo_emb)
        sketch_emb = self.sketch_adaptor(sketch_emb)
        return self.fusenet(photo_emb, sketch_emb)


class ConvAdaptorNet(nn.Module):
    def __init__(self, in_n_filters, out_n_filters, 
                 in_height, in_width, hid_dim, out_dim):
        """Many of the early convolutional networks are too big 
        for us to collapse into a fully connected network. Let's
        shrink down the number of filters + add a max pool. (NCHW)

        :param in_n_filters: number of filters in input tensor
        :param out_n_filters: number of filters in output tensor
        :param in_height: size of height
        :param in_width: size of width
        :param mid_dim: number of dimensions in middle FC layer
        :param out_dim: number of dimensions in output embedding
        """
        super(ConvAdaptorNet, self).__init__()
        self.conv = nn.Conv2d(in_n_filters, out_n_filters, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(size=(2, 2), stride=(2, 2), dilation=(1, 1))
        in_dim = out_n_filters * in_height / 2 * in_width / 2
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


class FuseClassifier(nn.Module):
    def forward(self, e1, e2):
        e = cosine_similarity(e1, e2, dim=1).unsqueeze(1)
        return F.sigmoid(e)


def generator_size(sketch_emb_dir, train=True):
    categories = os.listdir(sketch_emb_dir)
    n_categories = len(categories)

    if train:
        categories = categories[:int(n_categories * 0.8)]
    else:
        categories = categories[int(n_categories * 0.8):]

    sketch_paths = [path for path in list_files(sketch_emb_dir, ext='npy') 
                   if os.path.dirname(path).split('/')[-1] in categories]

    return len(sketch_paths) * 2


class EmbeddingGenerator(object):
    """This data generator returns (photo, sketch, label) where label
    is 0 or 1: same object, different object.
 
    :param photo_emb_dir: pass to photo embedding directories
    :param sketch_emb_dir: pass to sketch embedding directories
    :param train: if True, return 80% of data; else return 20%.
    :param batch_size: number to return at a time
    :param use_cuda: if True, make CUDA compatible objects
    :param strict: if True, sketches of the same class but different photo are
                   treated as negatives. The good part of doing this is to really
                   pull apart exact photo sketch matches. The bad part is that
                   noise and diff photo same class are about the same. Ideally, 
                   we want to have noise + diff class be about the same, and have
                   same class diff photo and same class same photo near each other.
    :param random_seed: so that random shuffle is the same everytime
    """
    def __init__(self, photo_emb_dir, sketch_emb_dir, batch_size=32, train=True, 
                 use_cuda=False, strict=False, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.photo_emb_dir = photo_emb_dir
        self.sketch_emb_dir = sketch_emb_dir
        self.batch_size = batch_size
        self.train = train
        self.use_cuda = use_cuda
        self.strict = strict

    def gen_photo_from_sketch_filename(self, sketch_filename):
        """Overwrite this function for other purposes."""
        photo_filename = sketch_filename.split('-')[0] + '.npy'
        return photo_filename

    def make_generator(self):
        categories = os.listdir(self.sketch_emb_dir)
        n_categories = len(categories)

        categories = (categories[:int(n_categories * 0.8)] if self.train 
                      else categories[int(n_categories * 0.8):])
        photo_paths = [path for path in list_files(self.photo_emb_dir, ext='npy') 
                       if os.path.dirname(path).split('/')[-1] in categories]
        sketch_paths = [path for path in list_files(self.sketch_emb_dir, ext='npy') 
                       if os.path.dirname(path).split('/')[-1] in categories]

        sketch_paths_0 = copy.deepcopy(sketch_paths)
        sketch_paths_1 = copy.deepcopy(sketch_paths)
        sketch_paths_2 = copy.deepcopy(sketch_paths)

        random.shuffle(sketch_paths_0)
        random.shuffle(sketch_paths_1)
        random.shuffle(sketch_paths_2)

        n_paths = len(sketch_paths) * 2
        
        if self.strict:
            class_samples = np.random.choice(range(3), size=n_paths, p=[0.5, 0.25, 0.25])
        else:
            class_samples_0 = [0 for i in range(len(sketch_paths))]
            class_samples_1 = [1 for i in range(len(sketch_paths))]
            class_samples = class_samples_0 + class_samples_1
            random.shuffle(class_samples)

        class_0_i = 0
        class_1_i = 0
        class_2_i = 0

        photo_batch = None
        sketch_batch = None
        label_batch = None

        for i in range(n_paths):
            if class_samples[i] == 0:
                sketch_path = sketch_paths_0[class_0_i]
                sketch_filename = os.path.basename(sketch_path)
                sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
                photo_filename = self.gen_photo_from_sketch_filename(sketch_filename)
                photo_path = os.path.join(self.photo_emb_dir, sketch_folder, photo_filename)
                class_0_i += 1
            elif class_samples[i] == 1:
                sketch_path = sketch_paths_1[class_1_i]
                sketch_filename = os.path.basename(sketch_path)
                sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
                while True:
                    photo_path = np.random.choice(photo_paths)
                    photo_folder = os.path.dirname(photo_path).split('/')[-1]
                    if photo_folder != sketch_folder:
                        break
                class_1_i += 1
            else:
                sketch_path = sketch_paths_2[class_2_i]
                sketch_filename = os.path.basename(sketch_path)
                sketch_folder = os.path.dirname(sketch_path).split('/')[-1]
                matching_photo_filename = self.gen_photo_from_sketch_filename(sketch_filename)
                matching_photo_path = os.path.join(self.photo_emb_dir, sketch_folder, matching_photo_filename)             
                matching_photo_folder = os.path.dirname(matching_photo_path)

                while True:                        
                    photo_filename = np.random.choice(os.listdir(matching_photo_folder))
                    photo_path = os.path.join(matching_photo_folder, photo_filename)
                    if photo_filename != matching_photo_filename:
                        break
                class_2_i += 1

            photo = np.load(photo_path)
            sketch = np.load(sketch_path)
            label = int(class_samples[i] == 0)

            if photo_batch is None and sketch_batch is None:
                photo_batch = photo
                sketch_batch = sketch
                label_batch = [label]
            else:
                photo_batch = np.vstack((photo_batch, photo))
                sketch_batch = np.vstack((sketch_batch, sketch))
                label_batch.append(label)

            if photo_batch.shape[0] == self.batch_size:
                photo_batch = torch.from_numpy(photo_batch)
                sketch_batch = torch.from_numpy(sketch_batch)
                label_batch = np.array(label_batch)
                label_batch = torch.from_numpy(label_batch)

                if self.train:
                    photo_batch = Variable(photo_batch)
                    sketch_batch = Variable(sketch_batch)
                else:
                    photo_batch = Variable(photo_batch, volatile=True)
                    sketch_batch = Variable(sketch_batch, volatile=True)
                label_batch = Variable(label_batch, requires_grad=False)

                if self.use_cuda:
                    photo_batch = photo_batch.cuda()
                    sketch_batch = sketch_batch.cuda()
                    label_batch = label_batch.cuda()

                yield (photo_batch, sketch_batch, label_batch)
                
                photo_batch = None
                sketch_batch = None
                label_batch = None

        # return any remaining data
        if photo_batch is not None and sketch_batch is not None:
            photo_batch = torch.from_numpy(photo_batch)
            sketch_batch = torch.from_numpy(sketch_batch)
            label_batch = np.array(label_batch)
            label_batch = torch.from_numpy(label_batch)
            if train:
                photo_batch = Variable(photo_batch)
                sketch_batch = Variable(sketch_batch)
            else:
                photo_batch = Variable(photo_batch, volatile=True)
                sketch_batch = Variable(sketch_batch, volatile=True)
            label_batch = Variable(label_batch, requires_grad=False)

            if self.use_cuda:
                photo_batch = photo_batch.cuda()
                sketch_batch = sketch_batch.cuda()
                label_batch = label_batch.cuda()

            yield (photo_batch, sketch_batch, label_batch)


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)

    if checkpoint['type'] == EMBED_NET_TYPE:
        model = EmbedNet()
    elif checkpoint['type'] == CONV_EMBED_NET_TYPE:
        model = ConvEmbedNet()
    else:
        raise Exception('Unknown model type %d.' % checkpoint['type'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


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
    parser.add_argument('photo_emb_folder', type=str)
    parser.add_argument('sketch_emb_folder', type=str)
    parser.add_argument('out_folder', type=str)
    parser.add_argument('--convolutional', action='store_true', default=False,
                        help='If True, initialize ConvEmbedNet.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--strict', action='store_true', default=False,
                        help='if True, then consider a sketch of the same class but different photo as negative.')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    photo_emb_dir = args.photo_emb_folder
    sketch_emb_dir = args.sketch_emb_folder

    def reset_generators():
        train_generator = EmbeddingGenerator(photo_emb_dir, sketch_emb_dir,
                                             batch_size=args.batch_size, train=True, 
                                             strict=args.strict, use_cuda=args.cuda)
        test_generator = EmbeddingGenerator(photo_emb_dir, sketch_emb_dir, 
                                            batch_size=args.batch_size, train=False, 
                                            strict=args.strict, use_cuda=args.cuda)
        return train_generator.make_generator(), test_generator.make_generator()

    train_generator, test_generator = reset_generators()
    n_train = generator_size(sketch_emb_dir, train=True)
    n_test = generator_size(sketch_emb_dir, train=False)

    if args.convolutional:
        model = ConvEmbedNet()
        model_type = CONV_EMBED_NET_TYPE
    else:
        model = EmbedNet()
        model_type = EMBED_NET_TYPE

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()


    def train(epoch):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        model.train()
        batch_idx = 0
        
        while True:
            try:
                photos, sketches, labels = train_generator.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            loss_meter.update(loss.data[0], photos.size(0)) 
            
            if args.cuda:
                acc = accuracy_score(labels.cpu().data.numpy(),
                                     np.round(outputs.cpu().squeeze(1).data.numpy(), 0))
            else:
                acc = accuracy_score(labels.data.numpy(),
                                     np.round(outputs.squeeze(1).data.numpy(), 0))
            acc_meter.update(acc, photos.size(0))

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                      epoch, batch_idx * args.batch_size, n_train,
                      (100. * batch_idx * args.batch_size) / n_train,
                      loss_meter.avg, acc_meter.avg))


    def test(epoch):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        model.eval()
        batch_idx = 0
        
        while True:
            try:
                photos, sketches, labels = test_generator.next()
                batch_idx += 1
            except StopIteration:
                break
            
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            
            if args.cuda:
                acc = accuracy_score(labels.cpu().data.numpy(),
                                     np.round(outputs.cpu().squeeze(1).data.numpy(), 0))
            else:
                acc = accuracy_score(labels.data.numpy(),
                                     np.round(outputs.squeeze(1).data.numpy(), 0))
            
            acc_meter.update(acc, photos.size(0))
            loss_meter.update(loss.data[0], photos.size(0))

        print('Test Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}'.format(
              epoch, loss_meter.avg, acc_meter.avg))

        return acc_meter.avg


    print('')
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = test(epoch)

        train_generator, test_generator = reset_generators()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            'type': model_type,
        }, is_best, folder=args.out_folder)
