"""Format the json to be a particular way. This is to be called 
after dump.py."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json

from model import load_checkpoint
from model import cosine_similarity


def simplify_sketch(path):
    path = '_'.join(os.path.splitext(os.path.basename(path))[0].split('_')[1:])
    path = path.split('-')[-1]
    path = path.replace('_trial', '')
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    with open(args.input_path) as fp:
        data = json.load(fp)
    print('loaded json.')

    render_paths = [i['render'] for i in data]
    render_paths = set(render_paths)
    n_data = len(data)

    out_json = {}
    for path in render_paths:
        path = os.path.splitext(os.path.basename(path))[0]
        path = '_'.join(path.split('_')[2:])
        out_json[path] = {}
    
    for i, item in enumerate(data):
        render_key = os.path.splitext(os.path.basename(item['render']))[0]
        render_key = '_'.join(render_key.split('_')[2:])
        sketch_key = simplify_sketch(item['sketch'])
        out_json[render_key][sketch_key] = item['distance']
        print('Processed items [{}/{}]'.format(i + 1, n_data))

    with open(args.output_path, 'wb') as fp:
        json.dump(out_json, fp)
