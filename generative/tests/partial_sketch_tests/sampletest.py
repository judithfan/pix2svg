"""Given a partially rendered image, can we sample and 
complete the image? This does not have problems with 
differentiability and being blurry.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import csv
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms

# these will be used to wiggle.
sys.path.append('../..')
sys.path.append('../distribution_test')
sys.path.append('../multimodal_test')

from linerender import BresenhamRenderNet
from beamsearch import sample_endpoint_gaussian2d
from wiggletest import photo_preprocessing
from wiggletest import gen_endpoints_from_csv
from distribtest import cosine_similarity
from multimodaltest import load_checkpoint
from precompute_vgg import cnn_predict


def save_sketch(sketch, epoch, out_folder='./'):
    sketch = torch.cat((sketch, sketch, sketch), dim=1)
    sketch_np = sketch[0].numpy() * 255
    sketch_np = np.rollaxis(sketch_np, 0, 3)
    sketch_np = np.round(sketch_np, 0).astype(np.uint8)
    im = Image.fromarray(sketch_np)
    im.save(os.path.join(out_folder, 'sketch_{}.png'.format(epoch)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='path to the trained model file')
    parser.add_argument('out_folder', type=str,
                        help='where to save sketch')
    parser.add_argument('n_wiggle', type=int, 
                        help='number of segments to wiggle (from the end)')
    parser.add_argument('--beam_width', type=int, default=2, help='size of beam traversal.')
    # we may know the number of strokes left but we can't guarantee
    # that beam search will find something similar in efficiency.
    parser.add_argument('--n_segments', type=int, default=20, help='number of segments to draw')
    parser.add_argument('--n_samples', type=int, default=1000, help='number of points to draw at each iter')
    parser.add_argument('--stdev', type=float, default=40.0, help='standard deviation')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # load the VGG embedding net and the multimodal modal
    cnn = models.vgg19(pretrained=True)
    net = load_checkpoint(args.model_path, use_cuda=args.cuda)
    cnn.eval()
    net.eval()

    if args.cuda:
        cnn = cnn.cuda()
        net = net.cuda()

    for p in cnn.parameters():
        p.requires_grad = False

    for p in net.parameters():
        p.requires_grad = False

    # TODO: make this not hardcoded.
    photo_path = './data/n02691156_10168.jpg'

    # convert to torch object
    photo = Image.open(photo_path)
    photo = photo.convert('RGB')
    photo = photo_preprocessing(photo).unsqueeze(0)
    photo = Variable(photo, volatile=True)
    if args.cuda:
        photo = photo.cuda()
    photo = cnn_predict(photo, cnn)
    photo = net.photo_adaptor(photo)

    # here we are going to rip photo from its tape and recast it as a variable
    photo = Variable(photo.data)

    # HACK: 0-indexing for sketch_id inside CSV but 1-indexing for sketch_id in filename
    photo_csv_name = os.path.splitext(os.path.basename(photo_path))[0]
    endpoints = gen_endpoints_from_csv(photo_csv_name, 2)  # HARDCODED ID
    # HACK: coordinates are current in 640 by 480; reshape to 256
    #       AKA: transforms.Scale(256)
    endpoints[:, 0] = endpoints[:, 0] / 640 * 256
    endpoints[:, 1] = endpoints[:, 1] / 480 * 256

    # truncate endpoints
    endpoints = endpoints[:args.n_wiggle, :]
    n_endpoints = endpoints.shape[0]

    # -----------------------------------------------------------------
    # begin beam search.

    # start prepping variables for beam search
    # start our points at where we ended off
    x0 = endpoints[-1, 0]
    y0 = endpoints[-1, 1]

    x_beam_queue = np.ones(beam_width) * x0
    y_beam_queue = np.ones(beam_width) * y0

    x_beam_paths = np.zeros((beam_width, n_iters + 1))
    y_beam_paths = np.zeros((beam_width, n_iters + 1))
    pen_beam_paths = np.ones(beam_width, n_endpoints + args.n_segments) * 2

    x_beam_paths[:, :n_endpoints] = endpoints[:, 0]
    y_beam_paths[:, :n_endpoints] = endpoints[:, 1]
    # always draw for new segments
    pen_beam_paths[:, :n_endpoints] = endpoints[:, 2]


    def train(epoch):
        global x_beam_queue
        global y_beam_queue
        global x_beam_paths
        global y_beam_paths

        print('epoch [{}/{}]'.format(iter + 1, args.n_segments))

        for b in range(beam_width):
            print('- beam [{}/{}]'.format(b + 1, args.beam_width))
            # sample endpoints
            samples = sample_endpoint_gaussian2d(x_beam_queue[b], y_beam_queue[b],
                                                 std=args.stdev, size=args.n_samples,
                                                 min_x=0, max_x=256, min_y=0, max_y=256)
            x_samples, y_samples = samples[:, 0], samples[:, 1]
            print('-- sampled {} points'.format(n_samples))
            losses = torch.zeros((n_samples))

            # for each sample & render image
            for i in range(n_samples):
                x_list = copy.deepcopy(x_beam_paths[b])
                y_list = copy.deepcopy(y_beam_paths[b])
                pen_list = copy.deepcopy(pen_beam_paths[b])
                x_list[n_endpoints + epoch] = x_samples[i]
                y_list[n_endpoints + epoch] = y_samples[i]
                renderer = BresenhamRenderNet(y_list[:n_endpoints + epoch + 1],
                                              x_list[:n_endpoints + epoch + 1],
                                              pen_list=pen_list[:n_endpoints + epoch + 1],
                                              imsize=256, linewidth=5)
                # at this point, sketch is a (1, 1, 256, 256) object
                sketch = renderer.forward()
                sketch = 1 - sketch  # only for bresenhams
                sketch = Variable(sketch, volatile=True)
                if use_cuda:
                    sketch = sketch.cuda()

                # HACK: manually center crop to 224 by 224 from 256 by 256
                #       AKA transforms.CenterCrop(224)
                sketch = sketch[:, :, 16:240, 16:240]
                # HACK: given sketch 3 channels: RGB
                sketch = torch.cat((sketch, sketch, sketch), dim=1)
                # HACK: manually normalize each dimension
                #       AKA transforms.Normalize([0.485, 0.456, 0.406],
                #                                [0.229, 0.224, 0.225])
                sketch[:, 0] = (sketch[:, 0] - 0.485) / 0.229
                sketch[:, 1] = (sketch[:, 1] - 0.456) / 0.224
                sketch[:, 2] = (sketch[:, 2] - 0.406) / 0.225

                sketch = cnn_predict(sketch, cnn)
                sketch = net.sketch_adaptor(sketch)

                loss = 1 - cosine_similarity(photo, sketch, dim=1)
                losses[i] = float(loss.cpu().data.numpy()[0])
                
                if (i + 1) % 25 == 0:
                    print('--- calc loss for [{}/{}] samples'.format(i + 1, n_samples))

            if b == 0:
                beam_losses = losses.numpy()
                x_beam_samples = x_samples
                y_beam_samples = y_samples
            else:
                beam_losses = np.concatenate((beam_losses, losses.numpy()))
                x_beam_samples = np.concatenate((x_beam_samples, x_samples))
                y_beam_samples = np.concatenate((y_beam_samples, y_samples))

        top_ii = np.argsort(beam_losses)[:beam_width]
        _x_beam_paths = copy.deepcopy(x_beam_paths)
        _y_beam_paths = copy.deepcopy(y_beam_paths)

        for b in range(beam_width):
            parent = top_ii[b] // n_samples
            _x_beam_paths[b][n_endpoints + epoch] = x_beam_samples[top_ii[b]]
            _y_beam_paths[b][n_endpoints + epoch] = y_beam_samples[top_ii[b]]

        x_beam_paths = _x_beam_paths
        y_beam_paths = _y_beam_paths
        x_beam_queue = np.array([x_beam_samples[top_ii[b]] for b in range(beam_width)])
        y_beam_queue = np.array([y_beam_samples[top_ii[b]] for b in range(beam_width)])

        best_ii = top_ii[0] // n_samples
        x_list = x_beam_paths[best_ii][:n_endpoints + epoch + 1]
        y_list = y_beam_paths[best_ii][:n_endpoints + epoch + 1]
        pen_list = pen_beam_paths[best_ii][:n_endpoints + epoch + 1]
        print('- updated global beam variables...')

        top_renderer = BresenhamRenderNet(y_list, x_list, pen_list=pen_list, 
                                          imsize=256, linewidth=5)
        top_sketch = top_renderer.forward()
        top_sketch = 1 - top_sketch
        print('- generated top sketch | loss: {}'.format(beam_losses[best_ii]))

        return top_sketch


print('training beam model...')
for iter in range(args.n_segments):
    sketch = train(iter)
    save_sketch(sketch, iter, out_folder=args.out_folder)

print('saving final sketch...')
save_sketch(sketch, 'final', out_folder=args.out_folder)
