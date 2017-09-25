"""Neural net to learn a transformation from differentiable 
sketch to sketch."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys;
sys.path.append('../multimodal_test')
from multimodal_test import generator_size
from multimodal_test import embedding_generator


class Sketch2SketchNet(nn.Module):
    def __init__(self, in_dim):
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        return self.fc2(x)


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


def save_checkpoint(state, is_best, outdir, filename='checkpoint'):
    checkpoint_path = os.path.join(outdir, '{}.pth.tar'.format(filename))
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, 
                        os.path.join(outdir, '{}.best.pth.tar'.format(filename)))


def load_checkpoint(file_path, use_cuda=False):
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path, 
                                map_location=lambda storage, location: storage)

    model = Sketch2SketchNet(4096)
    model.load_state_dict(checkpoint['state_dict'])

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_folder', type=str, help='where to save trained model.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    gt_sketch_emb_dir = '/home/wumike/full_sketchy_embeddings/sketches'
    diff_sketch_emb_dir = '/home/wumike/full_sketchy_embeddings/diff_sketches'

    def reset_generators():
        train_generator = embedding_generator(gt_sketch_emb_dir, diff_sketch_emb_dir, imsize=256, 
                                              batch_size=args.batch_size, train=True, use_cuda=args.cuda)
        test_generator = embedding_generator(gt_sketch_emb_dir, diff_sketch_emb_dir, imsize=256, 
                                             batch_size=args.batch_size, train=True, use_cuda=args.cuda)
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    n_train = generator_size(gt_sketch_emb_dir, train=True)
    n_test = generator_size(gt_sketch_emb_dir, train=False)

    model = Sketch2SketchNet(4096)
    if args.cuda:
        model.cuda()


    def train(epoch):
        loss_meter = AverageMeter()

        model.train()
        batch_idx = 0
        
        while True:
            try:
                gt_sketches, diff_sketches, labels = train_generator.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            
            diff_sketches = model(diff_sketches)
            loss = torch.norm(gt_sketches - diff_sketches, p=2)
            loss_meter.update(loss.data[0], len(gt_sketches))
            
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * args.batch_size, n_train,
                      (100. * batch_idx * args.batch_size) / n_train,
                      loss_meter.avg))

    def test(epoch):
        loss_meter = AverageMeter()

        model.train()
        batch_idx = 0
        
        while True:
            try:
                gt_sketches, diff_sketches, labels = train_generator.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            
            diff_sketches = model(diff_sketches)
            loss = torch.norm(gt_sketches - diff_sketches, p=2)
            loss_meter.update(loss.data[0], len(gt_sketches))
            
        print('Test Epoch: {}\tLoss: {:.6f}'.format(
              epoch, loss_meter.avg))

        return loss_meter.avg


    print('')
    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss = test(epoch)

        train_generator, test_generator = reset_generators()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
        }, is_best, folder=args.out_folder)
