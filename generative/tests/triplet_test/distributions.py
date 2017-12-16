from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import cPickle
from collections import defaultdict
from datasets import CATEGORY_LOOKUP, CATEGORY_TO_INSTANCE_DICT

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def invert_many_to_one(d):
    d_invert = defaultdict(lambda: [])
    for key, value in d.iteritems():
        d_invert[value].append(key)
    return d_invert


def invert_one_to_one(d):
    d_invert = {}
    for key, value in d.iteritems():
        d_invert[value] = key
    return d_invert


def flatten_list(lists):
    flat_list = []
    for lst in lists:
        flat_list += lst
    return flat_list


def get_category_from_target(target):
    return os.path.splitext(target)[0].split('_')[-1]


def plot_quartet(close_sketch_close_target, close_sketch_close_distractor,
                 far_sketch_far_target, far_sketch_far_distractor,
                 close_sketch_far_target, close_sketch_far_distractor,
                 far_sketch_close_target, far_sketch_close_distractor,
                 save_path='./plot.png'):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4)
    sns.kdeplot(close_sketch_close_target, shade=False, ax=ax0, label='target')
    sns.kdeplot(close_sketch_close_distractor, shade=False, ax=ax0,  label='distractors')
    sns.kdeplot(far_sketch_far_target, shade=False, ax=ax1, label='target')
    sns.kdeplot(far_sketch_far_distractor, shade=False, ax=ax1, label='distractors')
    sns.kdeplot(close_sketch_far_target, shade=False, ax=ax2, label='target')
    sns.kdeplot(close_sketch_far_distractor, shade=False, ax=ax2, label='distractors')
    sns.kdeplot(far_sketch_close_target, shade=False, ax=ax3, label='target')
    sns.kdeplot(far_sketch_close_distractor, shade=False, ax=ax3, label='distractors')
    ax0.set_title('Close Sketch with Close Render')
    ax1.set_title('Far Sketch with Far Render')
    ax2.set_title('Close Sketch with Far Render')
    ax3.set_title('Far Sketch with Close Render')
    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax0.set_xlim(-1, 1)
    ax1.set_xlim(-1, 1)
    ax2.set_xlim(-1, 1)
    ax3.set_xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', type=str, help='path to unformatted json path')
    parser.add_argument('pickle_path', type=str, help='path to pickle files')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir('./results'):
        os.mkdir('./results')

    with open(args.json_path) as fp:
        dump = json.load(fp)
        dictionary = {}
        for item in dump:
            item_key = '%s+%s' % (os.path.basename(item['sketch']),
                                  get_category_from_target(os.path.basename(item['render'])))
            dictionary[item_key] = item['distance']

    with open(args.pickle_path) as fp:
        data = cPickle.load(fp)
        target2sketch = data['target2sketch']
        distractor2sketch = data['distractor2sketch']
        target2distractors = data['target2distractors']
        target2condition = data['target2condition']

    sketch2target = invert_one_to_one(target2sketch)
    sketch2distractor = invert_many_to_one(distractor2sketch)

    # find targets that have been used in far and close conditions
    instance2closepairs = defaultdict(lambda: [])
    instance2farpairs = defaultdict(lambda: [])

    for target, condition in target2condition.iteritems():
        sketch = target2sketch[target]
        instance = os.path.splitext(target)[0].split('_')[-1]
        distractors = target2distractors[target]
        example = [sketch, target, distractors]
        if condition == 'closer':
            instance2closepairs[instance].append(example)
        else:
            instance2farpairs[instance].append(example)

    close_sketch_close_target_db = {}
    close_sketch_close_distractor_db = {}
    close_sketch_far_target_db = {}
    close_sketch_far_distractor_db = {}
    far_sketch_far_target_db = {}
    far_sketch_far_distractor_db = {}
    far_sketch_close_target_db = {}
    far_sketch_close_distractor_db = {}

    for instance in CATEGORY_LOOKUP.keys():
        close_examples = instance2closepairs[instance]
        far_examples = instance2farpairs[instance]

        _close_sketch_close_target_db = {}
        _close_sketch_close_distractor_db = {}
        _close_sketch_far_target_db = {}
        _close_sketch_far_distractor_db = {}
        _far_sketch_far_target_db = {}
        _far_sketch_far_distractor_db = {}
        _far_sketch_close_target_db = {}
        _far_sketch_close_distractor_db = {}

        for close_example in close_examples:
            close_sketch = close_example[0]
            close_target = close_example[1]  # same as far_target
            close_distractor1 = close_example[2][0]
            close_distractor2 = close_example[2][1]
            close_distractor3 = close_example[2][2]

            for far_example in far_examples:
                far_sketch = far_example[0]
                far_target = far_example[1]  # same as close_target
                far_distractor1 = far_example[2][0]
                far_distractor2 = far_example[2][1]
                far_distractor3 = far_example[2][2]

                close_sketch_close_target = dictionary['%s+%s' % (close_sketch, get_category_from_target(close_target))]
                close_sketch_close_distractor1 = dictionary['%s+%s' % (close_sketch, get_category_from_target(close_distractor1))]
                close_sketch_close_distractor2 = dictionary['%s+%s' % (close_sketch, get_category_from_target(close_distractor2))]
                close_sketch_close_distractor3 = dictionary['%s+%s' % (close_sketch, get_category_from_target(close_distractor3))]

                close_sketch_far_target = dictionary['%s+%s' % (close_sketch, get_category_from_target(far_target))]
                close_sketch_far_distractor1 = dictionary['%s+%s' % (close_sketch, get_category_from_target(far_distractor1))]
                close_sketch_far_distractor2 = dictionary['%s+%s' % (close_sketch, get_category_from_target(far_distractor2))]
                close_sketch_far_distractor3 = dictionary['%s+%s' % (close_sketch, get_category_from_target(far_distractor3))]

                far_sketch_far_target = dictionary['%s+%s' % (far_sketch, get_category_from_target(far_target))]
                far_sketch_far_distractor1 = dictionary['%s+%s' % (far_sketch, get_category_from_target(far_distractor1))]
                far_sketch_far_distractor2 = dictionary['%s+%s' % (far_sketch, get_category_from_target(far_distractor2))]
                far_sketch_far_distractor3 = dictionary['%s+%s' % (far_sketch, get_category_from_target(far_distractor3))]

                far_sketch_close_target = dictionary['%s+%s' % (far_sketch, get_category_from_target(close_target))]
                far_sketch_close_distractor1 = dictionary['%s+%s' % (far_sketch, get_category_from_target(close_distractor1))]
                far_sketch_close_distractor2 = dictionary['%s+%s' % (far_sketch, get_category_from_target(close_distractor2))]
                far_sketch_close_distractor3 = dictionary['%s+%s' % (far_sketch, get_category_from_target(close_distractor3))]

                assert close_sketch_close_target == close_sketch_far_target
                assert far_sketch_far_target == far_sketch_close_target

                _close_sketch_close_target_db['%s+%s' % (close_sketch, get_category_from_target(close_target))] = close_sketch_close_target
                _close_sketch_close_distractor_db['%s+%s' % (close_sketch, get_category_from_target(close_distractor1))] = close_sketch_close_distractor1
                _close_sketch_close_distractor_db['%s+%s' % (close_sketch, get_category_from_target(close_distractor2))] = close_sketch_close_distractor2
                _close_sketch_close_distractor_db['%s+%s' % (close_sketch, get_category_from_target(close_distractor3))] = close_sketch_close_distractor3
                _close_sketch_far_target_db['%s+%s' % (close_sketch, get_category_from_target(far_target))] = close_sketch_far_target
                _close_sketch_far_distractor_db['%s+%s' % (close_sketch, get_category_from_target(far_distractor1))] = close_sketch_far_distractor1
                _close_sketch_far_distractor_db['%s+%s' % (close_sketch, get_category_from_target(far_distractor2))] = close_sketch_far_distractor2
                _close_sketch_far_distractor_db['%s+%s' % (close_sketch, get_category_from_target(far_distractor3))] = close_sketch_far_distractor3
                _far_sketch_far_target_db['%s+%s' % (far_sketch, get_category_from_target(far_target))] = far_sketch_far_target
                _far_sketch_far_distractor_db['%s+%s' % (far_sketch, get_category_from_target(far_distractor1))] = far_sketch_far_distractor1
                _far_sketch_far_distractor_db['%s+%s' % (far_sketch, get_category_from_target(far_distractor2))] = far_sketch_far_distractor2
                _far_sketch_far_distractor_db['%s+%s' % (far_sketch, get_category_from_target(far_distractor3))] = far_sketch_far_distractor3
                _far_sketch_close_target_db['%s+%s' % (far_sketch, get_category_from_target(close_target))] = far_sketch_close_target
                _far_sketch_close_distractor_db['%s+%s' % (far_sketch, get_category_from_target(close_distractor1))] = far_sketch_close_distractor1
                _far_sketch_close_distractor_db['%s+%s' % (far_sketch, get_category_from_target(close_distractor2))] = far_sketch_close_distractor2
                _far_sketch_close_distractor_db['%s+%s' % (far_sketch, get_category_from_target(close_distractor3))] = far_sketch_close_distractor3

        close_sketch_close_target_db[instance] = _close_sketch_close_target_db
        close_sketch_close_distractor_db[instance] = _close_sketch_close_distractor_db
        close_sketch_far_target_db[instance] = _close_sketch_far_target_db
        close_sketch_far_distractor_db[instance] = _close_sketch_far_distractor_db
        far_sketch_far_target_db[instance] = _far_sketch_far_target_db
        far_sketch_far_distractor_db[instance] = _far_sketch_far_distractor_db
        far_sketch_close_target_db[instance] = _far_sketch_close_target_db
        far_sketch_close_distractor_db[instance] = _far_sketch_close_distractor_db


    for instance in CATEGORY_LOOKUP.keys():
        plot_quartet(close_sketch_close_target_db[instance].values(), 
                     close_sketch_close_distractor_db[instance].values(),
                     far_sketch_far_target_db[instance].values(), 
                     far_sketch_far_distractor_db[instance].values(),
                     close_sketch_far_target_db[instance].values(), 
                     close_sketch_far_distractor_db[instance].values(),
                     far_sketch_close_target_db[instance].values(), 
                     far_sketch_close_distractor_db[instance].values(),
                     save_path='./results/%s.png' % instance)

    for category, instances in CATEGORY_TO_INSTANCE_DICT.iteritems():
        plot_quartet(flatten_list([close_sketch_close_target_db[instance].values() for instance in instances]),
                     flatten_list([close_sketch_close_distractor_db[instance].values() for instance in instances]),
                     flatten_list([far_sketch_far_target_db[instance].values() for instance in instances]),
                     flatten_list([far_sketch_far_distractor_db[instance].values() for instance in instances]),
                     flatten_list([close_sketch_far_target_db[instance].values() for instance in instances]),
                     flatten_list([close_sketch_far_distractor_db[instance].values() for instance in instances]),
                     flatten_list([far_sketch_close_target_db[instance].values() for instance in instances]),
                     flatten_list([far_sketch_close_distractor_db[instance].values() for instance in instances]),
                     save_path='./results/%s.png' % category)

    plot_quartet(flatten_list([close_sketch_close_target_db[instance].values() for instance in CATEGORY_LOOKUP.keys()]),
                 flatten_list([close_sketch_close_distractor_db[instance].values() for instance in CATEGORY_LOOKUP.keys()]),
                 flatten_list([far_sketch_far_target_db[instance].values() for instance in CATEGORY_LOOKUP.keys()]),
                 flatten_list([far_sketch_far_distractor_db[instance].values() for instance in CATEGORY_LOOKUP.keys()]),
                 flatten_list([close_sketch_far_target_db[instance].values() for instance in CATEGORY_LOOKUP.keys()]),
                 flatten_list([close_sketch_far_distractor_db[instance].values() for instance in CATEGORY_LOOKUP.keys()]),
                 flatten_list([far_sketch_close_target_db[instance].values() for instance in CATEGORY_LOOKUP.keys()]),
                 flatten_list([far_sketch_close_distractor_db[instance].values() for instance in CATEGORY_LOOKUP.keys()]),
                 save_path='./results/all.png')
