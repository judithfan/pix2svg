"""Train the ranking model to hopefully learn distributions
that are pushed apart"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys

from model import SketchRankNet
from model import ranking_loss

from generator import EmbeddingGenerator

from utils import AverageMeter
from utils import save_checkpoint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_folder', type=str)
    parser.add_argument('sketch_emb_folder', type=str)
    parser.add_argument('noise_emb_folder', type=str)
    parser.add_argument('out_folder', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()


    def reset_generators():
        train_generator = EmbeddingGenerator(args.photo_emb_dir, args.sketch_emb_dir, 
                                             args.noise_emb_dir, batch_size=args.batch_size, 
                                             train=True, use_cuda=args.cuda)
        test_generator = EmbeddingGenerator(args.photo_emb_dir, args.sketch_emb_dir, 
                                            args.noise_emb_dir, batch_size=args.batch_size, 
                                            train=False, use_cuda=args.cuda)
        return train_generator, test_generator


    train_generator, test_generator = reset_generators()
    train_generator = train_generator.make_generator()
    test_generator = test_generator.make_generator()
    n_train = train_generator.size
    n_test = test_generator.size

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
                 sketches_diff_class, noises) = train_generator.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            outputs = model(photos, sketches_same_photo, sketches_same_class, 
                            sketches_diff_class, noises)
            loss = ranking_loss(outputs)
            loss_meter.update(loss.data[0], len(photos)) 
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * args.batch_size, n_train,
                      (100. * batch_idx * args.batch_size) / n_train, 
                      loss_meter.avg))


    def test(epoch):
        loss_meter = AverageMeter()
        model.eval()
        batch_idx = 0

        while True:
            try:
                (photos, sketches_same_photo, sketches_same_class, 
                 sketches_diff_class, noises) = test_generator.next()
                batch_idx += 1
            except StopIteration:
                break

            outputs = model(photos, sketches_same_photo, sketches_same_class, 
                            sketches_diff_class, noises)
            loss = ranking_loss(outputs)
            loss_meter.update(loss.data[0], len(photos)) 

        print('Test Epoch: {}\tLoss: {:.6f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg


    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss = test(epoch)
        
        train_generator, test_generator = reset_generators()
        train_generator = train_generator.make_generator()
        test_generator = test_generator.make_generator()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=args.out_folder)
