from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import json
import torch

from generators2 import ReferenceGame2EmbeddingGenerator

sys.path.append('..')
from triplet_test.model import load_checkpoint
from triplet_test.model import cosine_similarity


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', type=str, help='path to where to dump json')
    parser.add_argument('model_dir', type=str, help='path to trained model')
    parser.add_argument('--v96', action='store_true', default=False, help='use 96 game version')
    parser.add_argument('--model', type=str, help='conv_4_2|fc7', default='conv_4_2')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.v96 = '96' if args.v96 else ''
    assert args.model in ['conv_4_2', 'fc7']

    generator = ReferenceGame2EmbeddingGenerator(
        data_dir='/data/jefan/sketchpad_basic_fixedpose%s_%s' % (args.v96, args.model),
        use_cuda=args.cuda)
    examples = generator.make_generator() 
    print('Built generator.')

    model = load_checkpoint(args.model_dir, use_cuda=args.cuda)
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

        # pass sketch and render in VGG (fc7) and then get MM embeddings
        # this is the same for our ranking model (luckily)
        sketch_emb = model.sketch_adaptor(sketch_emb)
        render_emb = model.photo_adaptor(render_emb)

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


    print('\nWriting distances to {}.')
    with open(args.json_path, 'w') as fp:
        json.dump(dist_jsons, fp)
