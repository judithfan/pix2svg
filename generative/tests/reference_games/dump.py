from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import json

import torch
from generator import ReferenceGameGenerator
import torchvision.models as models

sys.path.append('../multimodal_test')
from multimodaltest import load_checkpoint

sys.path.append('../distribution_test')
from distribtest import cosine_similarity


def cnn_predict(x, cnn):
    x = cnn.features(x)
    x = x.view(x.size(0), -1)
    classifier = list(cnn.classifier)[:4]  # fc7 layer
    for i in range(len(classifier)):
        x = classifier[i](x)
    return x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sketch_dir', type=str, help='path to sketches')
    parser.add_argument('render_dir', type=str, help='path to renderings')
    parser.add_argument('json_path', type=str, help='path to where to dump json')
    parser.add_argument('model_dir', type=str, help='path to trained MM model')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    print('\nSketch Directory: {}'.format(args.sketch_dir))
    print('3D-Render Directory: {}'.format(args.render_dir))
    print('Trained Model Directory: {}\n'.format(args.model_dir))

    _generator = ReferenceGameGenerator(args.sketch_dir, args.render_dir, use_cuda=args.cuda)
    generator = _generator.make_generator() 
    print('Built generator.')

    # load VGG
    cnn = models.vgg19(pretrained=True)
    cnn.eval()
    if args.cuda:
        cnn.cuda()
    print('Loaded VGG.')

    # load multimodal model  
    model = load_checkpoint(args.model_dir, use_cuda=args.cuda)
    model.eval()
    if model.cuda:
        model.cuda()
    print('Loaded multimodal model.')

    dist_jsons = []
    count = 0

    while True:
        try:  # exhaust the generator to return all pairs
            sketch_path, sketch, render_path, render = generator.next()
        except StopIteration:
            break

        # pass sketch and render in VGG (fc7) and then get MM embeddings
        sketch_emb = cnn_predict(sketch, cnn)
        render_emb = cnn_predict(render, cnn)
        sketch_emb = model.sketch_adaptor(sketch_emb)
        render_emb = model.photo_adaptor(render_emb)
        # compute cosine similarity
        dist = cosine_similarity(render_emb, sketch_emb, dim=1)
        dist = float(dist.cpu().data.numpy()[0])
        
        dist_json = {'sketch': sketch_path,
                     'render': render_path,
                     'distance': dist}
        dist_jsons.append(dist_json)

        print('Compute Distance [{}/{}].'.format(count + 1, _generator.size))
        count += 1


    print('\nWriting distances to {}.')
    with open(args.json_path, 'w') as fp:
        json.dump(dist_jsons, fp)
