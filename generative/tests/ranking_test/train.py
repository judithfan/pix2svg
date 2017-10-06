"""Train the ranking model to hopefully learn distributions
that are pushed apart"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys

import torch
import torch.optim as optim

from model import SketchRankNet
from model import ranking_loss

from generator import RankingGenerator

from utils import AverageMeter
from utils import save_checkpoint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_dir', type=str)
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('noise_emb_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()


    def reset_generators():
        train_generator = RankingGenerator(args.photo_emb_dir, args.sketch_emb_dir, 
                                           args.noise_emb_dir, batch_size=args.batch_size, 
                                           train=True, use_cuda=args.cuda)
        test_generator = RankingGenerator(args.photo_emb_dir, args.sketch_emb_dir, 
                                          args.noise_emb_dir, batch_size=args.batch_size, 
                                          train=False, use_cuda=args.cuda)
        return train_generator, test_generator


    train_generator, test_generator = reset_generators()
    train_examples = train_generator.make_generator()
    test_examples = test_generator.make_generator()

    model = SketchRankNet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()


    def train(epoch):
        loss_meter = AverageMeter()
        model.train()
        batch_idx = 0

        while True:
            try:
                (photos, sketches_same_photo, sketches_same_class, 
                 sketches_diff_class, noises) = train_examples.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            outputs = model(photos, sketches_same_photo, sketches_same_class, 
                            sketches_diff_class, noises)
            loss = ranking_loss(outputs, use_cuda=args.cuda)
            loss_meter.update(loss.data[0], len(photos)) 
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * args.batch_size, train_generator.size,
                      (100. * batch_idx * args.batch_size) / train_generator.size, 
                      loss_meter.avg))

        return loss_meter.avg


    def test(epoch):
        loss_meter = AverageMeter()
        model.eval()
        batch_idx = 0

        while True:
            try:
                (photos, sketches_same_photo, sketches_same_class, 
                 sketches_diff_class, noises) = test_examples.next()
                batch_idx += 1
            except StopIteration:
                break

            outputs = model(photos, sketches_same_photo, sketches_same_class, 
                            sketches_diff_class, noises)
            loss = ranking_loss(outputs, use_cuda=args.cuda)
            loss_meter.update(loss.data[0], len(photos)) 

        print('Test Epoch: {}\tLoss: {:.6f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg


    best_loss = sys.maxint
    save_loss = np.zeros((args.epochs, 2))
    for epoch in range(1, args.epochs + 1):
        loss_tr = train(epoch)
        loss_te = test(epoch)
        
        train_generator, test_generator = reset_generators()
        train_examples = train_generator.make_generator()
        test_examples = test_generator.make_generator()

        is_best = loss_te < best_loss
        best_loss = min(loss_te, best_loss)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=args.out_dir)

        save_loss[epoch - 1, 0] = loss_tr
        save_loss[epoch - 1, 1] = loss_te
        # save the losses
        np.save(os.path.join(args.out_dir, 'save_loss.npy'), 
                save_loss[:epoch])
