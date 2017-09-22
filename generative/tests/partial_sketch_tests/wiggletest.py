"""Given a known sketch with its natural photo, we have 
extracted a bunch of fixed coordinates. We can wiggle all 
of them together (using our differentiable line renderer).
We want to make sure it doesn't go anywhere crazy. Small
perturbations are okay and to be expected.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import csv

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms

# these will be used to wiggle.
sys.path.append('../..')
sys.path.append('../distribution_test')
sys.path.append('../multimodal_test')
from linerender import SketchRenderNet
from distribtest import cosine_similarity
from multimodaltest import load_checkpoint
from precompute_vgg import cnn_predict


photo_preprocessing = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

# sketches are already size 224 and tensors
sketch_preprocessing = transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])


def gen_endpoints_from_csv(photo_name, sketch_id):
    csv_path = '/home/wumike/pix2svg/preprocessing/tiny/stroke_dataframe.csv'
    sketch_points = []
    with open(csv_path, 'rb') as fp:
        reader = csv.reader(fp)
        for row in reader:
            if row[-1] == photo_name and row[-2] == sketch_id:
                # we are going to ignore pen type (lifting for now)
                sketch_points.append([float(row[1]), float(row[2]), int(row[3]), int(row[4])])

    sketch_points = np.array(sketch_points)
    return sketch_points


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sketch_filename', type=str, 
                        help='must be a file inside full_sketchy_dataset/sketches/airplanes/*')
    parser.add_argument('model_path', type=str,
                        help='path to the trained model file')
    parser.add_argument('out_folder', type=str,
                        help='where to save sketch')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--fuzz', type=float, default=1.0)
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

    # load sketch endpoints
    sketch_endpoints = gen_endpoints_from_csv(photo_name, sketch_id)

    renderer = SketchRenderNet(sketch_endpoints[:, 0], sketch_endpoints[:, 1], 
                               sketch_endpoints[:, 2], imsize=224, fuzz=args.fuzz)
    optimizer = optim.Adam(renderer.parameters(), lr=args.lr)


    def train(epoch):
        renderer.train()
        optimizer.zero_grad()
        sketch = renderer()
        sketch = sketch_preprocessing(sketch)
        sketch = cnn_predict(sketch, cnn)
        sketch = net.sketch_adaptor(sketch)
        loss = 1 - cosine_similarity(photo, sketch, dim=1)
        loss.backward()
        optimizer.step()
        if epoch % args.log_interval == 0:
            print('Train Epoch: {} \tCosine Distance: {:.6f}'.format(epoch, loss.data[0]))


    for i in range(args.epochs):
        train(i)

    sketch = renderer()
    plt.matshow(sketch[0][0].data.numpy())
    plt.savefig(os.path.join(args.out_folder, 'output.png'))
