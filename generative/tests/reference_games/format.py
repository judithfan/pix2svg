"""Format the json to be a particular way."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json


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
        out_json[os.path.basename(path)] = {}
    
    for i, item in enumerate(data):
        render_key = os.path.basename(item['render'])
        sketch_key = os.path.basename(item['sketch'])
        out_json[render_key][sketch_key] = item['distance']
        print('Processed items [{}/{}]'.format(i + 1, n_data))

    with open(args.output_path, 'wb') as fp:
        json.dump(out_json, fp)

