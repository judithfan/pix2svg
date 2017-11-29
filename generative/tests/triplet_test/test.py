from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from train import AverageMeter
from sklearn.metrics import accuracy_score

from datasets import ContextFreePreloadedGenerator as Generator
from model import load_checkpoint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to where model is stored')
    parser.add_argument('--layer', type=str, help='conv_4_2|fc7', default='conv_4_2')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.layer in ['conv_4_2', 'fc7']

    # note how we are not using the augmented dataset since at test time,
    # we don't care about how it does on cropped data.
    generator = Generator(train=False, batch_size=1, use_cuda=args.cuda, closer_only=False,
                          data_dir='/data/jefan/sketchpad_basic_fixedpose96_%s' % args.layer)
    examples = generator.make_generator()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    
    if args.cuda:
        model.cuda()


    def test():
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
                photos, sketches, labels, categories, instances = examples.next()
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

        print('Test Loss: {:.6f} ({:.3f}|{:.3f}|{:.3f})\tAcc: ({:.2f}|{:.2f}|{:.2f})'.format(
              loss_meter.avg, embedding_loss_meter.avg, category_loss_meter.avg, instance_loss_meter.avg, 
              embedding_acc_meter.avg, category_acc_meter.avg, instance_acc_meter.avg))


    test()
