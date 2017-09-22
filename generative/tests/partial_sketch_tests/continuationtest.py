"""Given a known sketch with its natural photo, we can
keep X% of its strokes, and try to generate the rest by 
sampling. 
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

import torchvision.transforms as transforms
import torchvision.models as models

sys.path.append('../..')
from linerender import BresenhamRenderNet
from beamsearch import sample_endpoint_gaussian2d
from wiggletest import photo_preprocessing, sketch_preprocessing
from wiggletest import gen_endpoints_from_csv


def truncate_sketch(endpoints, fraction):
    n_strokes = max(endpoints[:, 2])
    n_strokes = int(n_strokes * fraction)
    endpoints = endpoints[endpoints[:, 2] <= n_strokes]
    return endpoints


def save_sketch(sketch, epoch, out_folder='./'):
    sketch = sketch.int()
    sketch = torch.cat((sketch, sketch, sketch), dim=1)
    sketch = (1 - sketch) * 255
    sketch_np = np.rollaxis(sketch.numpy()[0], 0, 3).astype('uint8')
    im = Image.fromarray(sketch_np)
    im.save(os.path.join(out_folder, 'sketch_{}.png'.format(epoch)))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sketch_filename', type=str, 
                        help='must be a file inside full_sketchy_dataset/sketches/airplanes/*')
    parser.add_argument('model_path', type=str,
                        help='path to the trained model file')
    parser.add_argument('out_folder', type=str,
                        help='where to save sketch')
    parser.add_argument('--fraction', type=float, default=0.75,
                        help='fraction of strokes to keep')
    
    # we may know the number of strokes left but we can't guarantee
    # that beam search will find something similar in efficiency.
    parser.add_argument('--n_segments', type=int, default=20)
    # more beam search parameters
    parser.add_argument('--beam_width', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--stdev', type=float, default=10.0)

    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    net = load_checkpoint(args.model_path)
    cnn = models.vgg19(pretrained=True)
    cnn.eval()
    net.eval()

    if args.cuda:
        cnn.cuda()
        net.cuda()

    # parse out the photo filename and sketch id from the user input
    sketch_name, sketch_ext = os.path.splitext(args.sketch_filename)
    photo_name = sketch_name.split('-')[0]
    sketch_id = str(sketch_name.split('-')[1])
    photo_filename = photo_name + '.jpg'

    # get photo image
    photo_path = os.path.join('/home/jefan/full_sketchy_dataset/photos/airplane', 
                              photo_filename)
    # convert to torch object
    photo = photo = Image.open(photo_path)
    photo = photo.convert('RGB')
    photo = photo_preprocessing(photo).unsqueeze(0)
    photo = Variable(photo, volatile=True)
    if args.cuda:
        photo.cuda()
    photo = cnn_predict(photo, cnn)
    photo = net.photo_adaptor(photo)

    # load sketch endpoints (we also have to think about the pen movements)
    sketch_endpoints = gen_endpoints_from_csv(photo_name, sketch_id)

    # only take the first N strokes 
    sketch_endpoints = truncate_sketch(sketch_endpoints, args.fraction)
    n_endpoints = sketch_endpoints.shape[0]

    # start prepping variables for beam search
    # start our points at where we ended off
    x0 = sketch_endpoints[-1, 0]
    y0 = sketch_endpoints[-1, 1]

    x_beam_queue = np.ones(args.beam_width) * x0
    y_beam_queue = np.ones(args.beam_width) * y0

    x_beam_paths = np.zeros((beam_width, n_endpoints + args.n_segments))
    y_beam_paths = np.zeros((beam_width, n_endpoints + args.n_segments))
    pen_beam_paths = np.ones(beam_width, n_endpoints + args.n_segments) * 2

    x_beam_paths[:, :n_endpoints] = sketch_endpoints[:, 0]
    y_beam_paths[:, :n_endpoints] = sketch_endpoints[:, 1]
    # always draw for new segments
    pen_beam_paths[:, :n_endpoints] = sketch_endpoints[:, 2]


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
                                                 min_x=0, max_x=224, min_y=0, max_y=224)
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
                renderer = BresenhamRenderNet(x_list[:n_endpoints + epoch + 1], 
                                              y_list[:n_endpoints + epoch + 1],
                                              pen_list=pen_list[:n_endpoints + epoch + 1],
                                              imsize=imsize, linewidth=5)
                sketch = renderer.forward()
                sketch = torch.cat((sketch, sketch, sketch), dim=1)
                sketch = Variable(sketch, volatile=True)
                if use_cuda:
                    sketch = sketch.cuda()

                sketch = sketch_preprocessing(sketch)
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

        top_renderer = BresenhamRenderNet(x_list, y_list, pen_list=pen_list, 
                                          imsize=imsize, linewidth=5)
        top_sketch = top_renderer.forward()
        print('- generated top sketch | loss: {}'.format(beam_losses[best_ii]))

        return top_sketch


print('training beam model...')
for iter in range(args.n_segments):
    sketch = train(iter)
    save_sketch(sketch, iter, out_folder=args.out_folder)

print('saving final sketch...')
save_sketch(sketch, 'final', out_folder=args.out_folder)
