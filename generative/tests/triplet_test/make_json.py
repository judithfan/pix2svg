from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import json
import torch

from datasets import ReferenceGamePreloadedGenerator
from model import load_checkpoint
from model import cosine_similarity


def _flatten(self, x):
    return x.view(x.size(0), -1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', type=str, help='path to where to dump json')
    parser.add_argument('model_path', type=str, help='path to trained model')
    parser.add_argument('--data_dir', type=str, help='path to precomputed embeddings', default='/data/jefan/sketchpad_basic_fixedpose96_conv_4_2')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--raw', action='store_true', default=False, help='compute raw distances without applying adaptor (e.g., for getting unadapted early layer distances)')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    generator = ReferenceGamePreloadedGenerator(
        data_dir=args.data_dir,
        use_cuda=args.cuda)
    examples = generator.make_generator() 

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if model.cuda:
        model.cuda()
        
    dist_jsons = []
    count = 0

    while True:
        try:  # exhaust the generator to return all pairs. this loads 1 tuple at a time.
            sketch_emb_path, sketch_emb, render_emb_path, render_emb = examples.next()
        except StopIteration:
            break

        if not args.raw:
            # pass sketch and render in VGG (fc7) and then get MM embeddings
            # this is the same for our ranking model (luckily)
            sketch_emb = model.sketch_adaptor(sketch_emb)
            render_emb = model.photo_adaptor(render_emb)
        else:
            sketch_emb = _flatten(sketch_emb)
            render_emb = _flatten(render_emb)

        # compute cosine similarity
        render_emb = render_emb - torch.mean(render_emb, dim=1, keepdim=True)
        sketch_emb = sketch_emb - torch.mean(sketch_emb, dim=1, keepdim=True)
        dist = cosine_similarity(render_emb, sketch_emb, dim=1)
        
        dist = float(dist.cpu().data.numpy()[0])
        dist_json = {'sketch': sketch_emb_path,
                     'render': render_emb_path,
                     'distance': dist}
        dist_jsons.append(dist_json)

        print('Compute Distance [{}/{}].'.format(count + 1, generator.size))
        count += 1


    print('\nWriting distances to file.')
    with open(args.json_path, 'w') as fp:
        json.dump(dist_jsons, fp)
