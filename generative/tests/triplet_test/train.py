from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score

from model import EmbedNet
from model import save_checkpoint
from datasets import ContextFreePreloadedGenerator as Generator


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--photo_augment', action='store_true', default=False)
    parser.add_argument('--match_weight', type=float, default=1.0)
    parser.add_argument('--category_weight', type=float, default=1.0)
    parser.add_argument('--instance_weight', type=float, default=1.0)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.photo_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose96_photo_augmented_conv_4_2'
    else:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose96_conv_4_2'

    def reset_generators():
        train_generator = Generator(train=True, batch_size=args.batch_size, use_cuda=args.cuda, data_dir=data_dir)
        test_generator = Generator(train=False, batch_size=args.batch_size, use_cuda=args.cuda, data_dir=data_dir)
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    train_examples = train_generator.make_generator()
    test_examples = test_generator.make_generator()

    model = EmbedNet()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)


    def train(epoch):
        loss_meter = AverageMeter()
        embedding_loss_meter = AverageMeter()
        category_loss_meter = AverageMeter()
        instance_loss_meter = AverageMeter()
        embedding_acc_meter = AverageMeter()
        category_acc_meter = AverageMeter()
        instance_acc_meter = AverageMeter()

        model.train()
        batch_idx = 0

        while True: 
            try:
                photos, sketches, labels, categories, instances = train_examples.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            embedding_outputs, category_outputs, instance_outputs = model(photos, sketches)
            embedding_loss = F.binary_cross_entropy(embedding_outputs.squeeze(1), labels.float())
            category_loss = F.cross_entropy(category_outputs, categories)
            instance_loss = F.cross_entropy(instance_outputs, instances)
            loss = args.match_weight * embedding_loss \
                   + args.category_weight * category_loss \
                   + args.instance_weight * instance_loss
            
            embedding_loss_meter.update(embedding_loss.data[0], photos.size(0))
            category_loss_meter.update(category_loss.data[0], photos.size(0))
            instance_loss_meter.update(instance_loss.data[0], photos.size(0))
            loss_meter.update(loss.data[0], photos.size(0)) 

            # break tape and calculate accuracy
            labels_np = labels.cpu().data.numpy()
            embedding_outputs_np = np.round(embedding_outputs.cpu().squeeze(1).data.numpy(), 0)
            embedding_acc = accuracy_score(labels_np, embedding_outputs_np)
            embedding_acc_meter.update(embedding_acc, photos.size(0))

            categories_np = categories.cpu().data.numpy()
            category_outputs_np = np.argmax(category_outputs.cpu().data.numpy(), axis=1)
            category_acc = accuracy_score(categories_np, category_outputs_np)
            category_acc_meter.update(category_acc, photos.size(0))

            instances_np = instances.cpu().data.numpy()
            instance_outputs_np = np.argmax(instance_outputs.cpu().data.numpy(), axis=1)
            instance_acc = accuracy_score(instances_np, instance_outputs_np)
            instance_acc_meter.update(instance_acc, photos.size(0))

            # compute gradients + take a step
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} ({:.3f}|{:.3f}|{:.3f})\tAcc: ({:.2f}|{:.2f}|{:.2f})'.format(
                  epoch, batch_idx * args.batch_size, train_generator.size,
                  (100. * batch_idx * args.batch_size) / train_generator.size,
                  loss_meter.avg, embedding_loss_meter.avg, category_loss_meter.avg, instance_loss_meter.avg, 
                  embedding_acc_meter.avg, category_acc_meter.avg, instance_acc_meter.avg))


    def test(epoch):
        loss_meter = AverageMeter()
        embedding_loss_meter = AverageMeter()
        category_loss_meter = AverageMeter()
        instance_loss_meter = AverageMeter()
        embedding_acc_meter = AverageMeter()
        category_acc_meter = AverageMeter()
        instance_acc_meter = AverageMeter()

        model.eval()
        batch_idx = 0

        while True:
            try:
                photos, sketches, labels, categories, instances = test_examples.next()
                batch_idx += 1
            except StopIteration:
                break
            
            embedding_outputs, category_outputs, instance_outputs = model(photos, sketches)
            embedding_loss = F.binary_cross_entropy(embedding_outputs.squeeze(1), labels.float())
            category_loss = F.cross_entropy(category_outputs, categories)
            instance_loss = F.cross_entropy(instance_outputs, instances)
            loss = embedding_loss + category_loss + instance_loss
            
            embedding_loss_meter.update(embedding_loss.data[0], photos.size(0))
            category_loss_meter.update(category_loss.data[0], photos.size(0))
            instance_loss_meter.update(instance_loss.data[0], photos.size(0))
            loss_meter.update(loss.data[0], photos.size(0)) 

            labels_np = labels.cpu().data.numpy()
            embedding_outputs_np = np.round(embedding_outputs.cpu().squeeze(1).data.numpy(), 0)
            embedding_acc = accuracy_score(labels_np, embedding_outputs_np)
            embedding_acc_meter.update(embedding_acc, photos.size(0))

            categories_np = categories.cpu().data.numpy()
            category_outputs_np = np.argmax(category_outputs.cpu().data.numpy(), axis=1)
            category_acc = accuracy_score(categories_np, category_outputs_np)
            category_acc_meter.update(category_acc, photos.size(0))

            instances_np = instances.cpu().data.numpy()
            instance_outputs_np = np.argmax(instance_outputs.cpu().data.numpy(), axis=1)
            instance_acc = accuracy_score(instances_np, instance_outputs_np)
            instance_acc_meter.update(instance_acc, photos.size(0))

        print('Test Epoch: {}\tLoss: {:.6f} ({:.3f}|{:.3f}|{:.3f})\tAcc: ({:.2f}|{:.2f}|{:.2f})'.format(
              epoch, loss_meter.avg, embedding_loss_meter.avg, category_loss_meter.avg, instance_loss_meter.avg, 
              embedding_acc_meter.avg, category_acc_meter.avg, instance_acc_meter.avg))

        summed_acc = sum((embedding_acc_meter.avg, category_acc_meter.avg, instance_acc_meter.avg))
        return summed_acc


    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = test(epoch)

        train_generator, test_generator = reset_generators()
        train_examples = train_generator.make_generator()
        test_examples = test_generator.make_generator()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_summed_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=args.out_dir)   
