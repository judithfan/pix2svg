from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cPickle

from referenceutils2 import EntityGenerator


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('generator', type=str, help='cross|intra|entity')
    parser.add_argument('--model', type=str, help='conv_4_2|fc7', default='conv_4_2')
    parser.add_argument('--closer', action='store_true', help='if True, include only closer examples')
    parser.add_argument('--photo_augment', action='store_true')
    parser.add_argument('--sketch_augment', action='store_true')
    args = parser.parse_args()
    assert args.generator in ['cross', 'intra', 'entity']
    assert args.model in ['conv_4_2', 'fc7']
    
    if args.photo_augment and args.sketch_augment:
        raise Exception('Cannot pass both photo_augment and sketch_augment')
    if args.photo_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose_augmented2_%s' % args.model
    elif args.sketch_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose_augmented_%s' % args.model
    else:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose_%s' % args.model

    if args.generator == 'cross':
        generator = ThreeClassGenerator(closer_only=args.closer, data_dir=data_dir)
    elif args.generator == 'intra':
        generator = FourClassGenerator(closer_only=args.closer, data_dir=data_dir)
    elif args.generator == 'entity':
        generator = EntityGenerator(closer_only=args.closer, data_dir=data_dir)

    pickle_name = ('preloaded_%s_closer.pkl' % args.generator 
                   if args.closer else 'preloaded_%s_all.pkl' % args.generator)
    with open(os.path.join(data_dir, pickle_name), 'wb') as fp:
        cPickle.dump({'cat2target': generator.cat2target, 
                      'target2sketch': generator.target2sketch,
                      'distractor2sketch': generator.distractor2sketch,
                      'target2distractors': generator.target2distractors,
                      'path2folder': generator.path2folder}, fp)

    print('Pickle saved.')
