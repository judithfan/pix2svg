from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score

from convmodel import EmbedNet
from convmodel import save_checkpoint

from referenceutils2 import ThreeClassGenerator, FourClassGenerator
from train import AverageMeter
from convmodel import load_checkpoint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to where model is stored')
    parser.add_argument('generator', type=str, help='cross|intra')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # choose the right generator
    assert args.generator in ['cross', 'intra']
    if args.generator == 'cross':
        Generator = ThreeClassGenerator
    elif args.generator == 'intra':
        Generator = FourClassGenerator

    # note how we are not using the augmented dataset since at test time,
    # we don't care about how it does on cropped data.
    generator = Generator(train=False, batch_size=args.batch_size, use_cuda=args.cuda,
                          data_dir='/data/jefan/sketchpad_basic_fixedpose_conv_4_2')
    examples = generator.make_generator()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()


    def test():
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        model.eval()
        batch_idx = 0

        while True:
            try:
                photos, sketches, labels = examples.next()
                batch_idx += 1
            except StopIteration:
                break
            
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            loss_meter.update(loss.data[0], photos.size(0))

            labels_np = labels.cpu().data.numpy()
            outputs_np = np.round(outputs.cpu().squeeze(1).data.numpy(), 0)
            acc = accuracy_score(labels_np, outputs_np)
            acc_meter.update(acc, photos.size(0))

        print('Loss: {:.6f}\tAcc: {:.6f}'.format(loss_meter.avg, acc_meter.avg))
        return acc_meter.avg


    test()