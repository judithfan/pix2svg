from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cPickle

from referenceutils2 import (EntityGenerator, 
                             ContextFreeGenerator,
                             ThreeClassGenerator,
                             FourClassGenerator,
                             ContextBalancedGenerator)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('generator', type=str, help='cross|intra|entity|context|balance')
    parser.add_argument('--model', type=str, help='conv_4_2|fc7', default='conv_4_2')
    parser.add_argument('--closer', action='store_true', help='if True, include only closer examples')
    parser.add_argument('--v96', action='store_true', default=False, help='use 96 game version')
    parser.add_argument('--photo_augment', action='store_true')
    parser.add_argument('--sketch_augment', action='store_true')
    args = parser.parse_args()
    args.v96 = '96' if args.v96 else ''
    assert args.generator in ['cross', 'intra', 'entity', 'context', 'balance']
    assert args.model in ['conv_4_2', 'fc7']
    
    if args.photo_augment and args.sketch_augment:
        raise Exception('Cannot pass both photo_augment and sketch_augment')
    if args.photo_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose%s_photo_augmented_%s' % (args.v96, args.model)
    elif args.sketch_augment:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose%s_sketch_augmented_%s' % (args.v96, args.model)
    else:
        data_dir = '/data/jefan/sketchpad_basic_fixedpose%s_%s' % (args.v96, args.model)

    if args.generator == 'cross':
        generator = ThreeClassGenerator(closer_only=args.closer, data_dir=data_dir)
    elif args.generator == 'intra':
        generator = FourClassGenerator(closer_only=args.closer, data_dir=data_dir)
    elif args.generator == 'entity':
        generator = EntityGenerator(closer_only=args.closer, data_dir=data_dir)
    elif args.generator == 'context':
        generator = ContextFreeGenerator(closer_only=args.closer, data_dir=data_dir)
    elif args.generator == 'balance':
        generator = ContextBalancedGenerator(closer_only=args.closer, data_dir=data_dir)

    pickle_name = ('preloaded_%s_closer.pkl' % args.generator 
                   if args.closer else 'preloaded_%s_all.pkl' % args.generator)
    with open(os.path.join(data_dir, pickle_name), 'wb') as fp:
        cPickle.dump({'cat2target': generator.cat2target, 
                      'target2sketch': generator.target2sketch,
                      'distractor2sketch': generator.distractor2sketch,
                      'target2distractors': generator.target2distractors,
                      'path2folder': generator.path2folder,
                      'target2condition': generator.target2condition}, fp)

    print('Pickle saved.')
