"""Augment a dataset by adding filtered out squares i.e. cover up
part of the image with a fixed size square around the image.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import csv
import copy
import shutil
import numpy as np
from glob import glob
from PIL import Image

from torch.autograd import Variable
import torchvision.transforms as transforms

preprocessing = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])


def alpha_composite(front, back):
    """Alpha composite two RGBA images.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object

    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result


def alpha_composite_with_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)


def load(path, transparent=False):
    image = Image.open(path)
    if transparent:
        image = alpha_composite_with_color(image)
    image = image.convert('RGB')
    return np.asarray(image)


def _obscure(image, filter_size, filter_loc, color):
    """Add a patch of a fixed color of pixels over part of the input 
    image and save this as a new sketch.
    
    :param image: numpy array
    :param filter_size: the height and width of filter
    :param filter_loc: (x-coordinate, y_coordinate) for center of filter
    :param color: int; will be used to build (R, G, B)
    """
    filter_x, filter_y = filter_loc
    image2 = np.ones_like(image) * color
    x_min = max(filter_x - filter_size, 0)
    x_max = min(filter_x + filter_size, image.shape[0])
    y_min = max(filter_y - filter_size, 0)
    y_max = min(filter_y + filter_size, image.shape[1])
    
    block = image[x_min:x_max, y_min:y_max, :]
    image2[x_min:x_max, y_min:y_max, :] = block
    return image2


def bbox(image, ignore_color):
    test = np.sum(image, axis=2) == ignore_color * 3
    a, b = np.where(~test)
    # x, y, w, h
    return min(a), min(b), max(a) - min(a), max(b) - min(b)


def obscure(image, color=255, filter_size=75):
    x, y, w, h = bbox(image, color)
    filter_size = min(min(w, h), filter_size)
    xi, yi = x, y
    crops = []
    while xi + filter_size <= x + w:
        while yi + filter_size <= y + h:
            center = (xi + filter_size // 2, yi + filter_size // 2)
            crop = _obscure(image, filter_size, center, color)
            crops.append(crop)
            yi += filter_size
        xi += filter_size
        yi = y
    return crops


def gen_crop_name(name, crop_ix):
    parts = name.split('_')
    parts[1] = parts[1] + '-crop%d' % crop_ix
    return '_'.join(parts)


def augment_by_sketch():
    data_dir = '/data/jefan/sketchpad_basic_fixedpose_augmented'

    with open(os.path.join(data_dir, 'incorrect_trial_paths_pilot2.txt')) as fp:
        bad_games = fp.readlines()
        bad_games = [i.replace('.png\n', '.png') for i in bad_games]
        new_bad_games = []

    with open(os.path.join(data_dir, 'sketchpad_basic_pilot2_group_data.csv')) as fp:
        reader = csv.reader(fp)
        csv_data = []
        new_csv_data = []
        for row in reader:
            csv_data.append(row)
        # open csv data and load data
        header = csv_data[0]
        csv_data = csv_data[1:]

    for ix, row in enumerate(csv_data):
        path = 'gameID_{id}_trial_{trial}.png'.format(id=row[1], trial=row[2])
        sketch = load(os.path.join(data_dir, 'sketch', path), transparent=True)
        crops = obscure(sketch, color=255, filter_size=75)

        for i, crop in enumerate(crops):
            im = Image.fromarray(crop)
            save_path = gen_crop_name(path, i)
            im.save(os.path.join(data_dir, 'sketch', save_path))

            if path in bad_games:
                new_bad_games.append(save_path.replace('.png', '.png\n'))

            for name in ['target', 'distractor1', 'distractor2', 'distractor3']:
                find_name = glob(os.path.join(data_dir, name, os.path.splitext(path)[0] + '_*.png'))
                find_name = [t for t in find_name if 'crop' not in t]
                assert len(find_name) == 1
                find_name = find_name[0]
                copy_name = gen_crop_name(os.path.basename(find_name), i)
                shutil.copy(find_name, os.path.join(os.path.dirname(find_name), copy_name))

            new_row = copy.deepcopy(row)
            new_row[1] = row[1] + '-crop%d' % i
            new_csv_data.append(new_row)

        print('Progress: [{}/{}] files finished augmentation regime.'.format(
            ix+1, len(csv_data)))

    with open(os.path.join(data_dir, 'incorrect_trial_paths_pilot2.txt'), 'a') as fp:
        fp.writelines(new_bad_games)

    print('Wrote new examples to incorrect_trial_paths_pilot2.txt file.')

    with open(os.path.join(data_dir, 'sketchpad_basic_pilot2_group_data.csv'), 'a') as fp:
        writer = csv.writer(fp)
        for row in new_csv_data:
            writer.writerow(row)

    print('Wrote new examples to sketchpad_basic_pilot2_group_data.csv file.')


def augment_by_photo():
    data_dir = '/data/jefan/sketchpad_basic_fixedpose_augmented2'

    with open(os.path.join(data_dir, 'incorrect_trial_paths_pilot2.txt')) as fp:
        bad_games = fp.readlines()
        bad_games = [i.replace('.png\n', '.png') for i in bad_games]
        new_bad_games = []

    with open(os.path.join(data_dir, 'sketchpad_basic_pilot2_group_data.csv')) as fp:
        reader = csv.reader(fp)
        csv_data = []
        new_csv_data = []
        for row in reader:
            csv_data.append(row)
        # open csv data and load data
        header = csv_data[0]
        csv_data = csv_data[1:]

    for ix, row in enumerate(csv_data):
        sketch_path = 'gameID_{id}_trial_{trial}.png'.format(id=row[1], trial=row[2])
        target_path = 'gameID_{id}_trial_{trial}_{target}.png'.format(id=row[1], trial=row[2], target=row[4])
        distractor1_path = 'gameID_{id}_trial_{trial}_{target}.png'.format(id=row[1], trial=row[2], target=row[12])
        distractor2_path = 'gameID_{id}_trial_{trial}_{target}.png'.format(id=row[1], trial=row[2], target=row[13])
        distractor3_path = 'gameID_{id}_trial_{trial}_{target}.png'.format(id=row[1], trial=row[2], target=row[14])

        target = load(os.path.join(data_dir, 'target', target_path))
        distractor1 = load(os.path.join(data_dir, 'distractor1', distractor1_path))
        distractor2 = load(os.path.join(data_dir, 'distractor2', distractor2_path))
        distractor3 = load(os.path.join(data_dir, 'distractor3', distractor3_path))
        target_crops = obscure(target, color=127, filter_size=50)
        distractor1_crops = obscure(distractor1, color=127, filter_size=50)
        distractor2_crops = obscure(distractor2, color=127, filter_size=50)
        distractor3_crops = obscure(distractor3, color=127, filter_size=50)
        crops = zip(target_crops, distractor1_crops, distractor2_crops, distractor3_crops)

        for i, target_crop in enumerate(target_crops):
            target_im = Image.fromarray(target_crop)
            target_crop_path = gen_crop_name(target_path, i)
            target_im.save(os.path.join(data_dir, 'target', target_crop_path))

            # for each target, randomly sample a distractor crop
            distractor1_crop = random.choice(distractor1_crops)
            distractor2_crop = random.choice(distractor2_crops)
            distractor3_crop = random.choice(distractor3_crops)

            distractor1_im = Image.fromarray(distractor1_crop)
            distractor2_im = Image.fromarray(distractor2_crop)
            distractor3_im = Image.fromarray(distractor3_crop)

            distractor1_crop_path = gen_crop_name(distractor1_path, i)
            distractor2_crop_path = gen_crop_name(distractor2_path, i)
            distractor3_crop_path = gen_crop_name(distractor3_path, i)

            distractor1_im.save(os.path.join(data_dir, 'distractor1', distractor1_path))
            distractor2_im.save(os.path.join(data_dir, 'distractor2', distractor2_path))
            distractor3_im.save(os.path.join(data_dir, 'distractor3', distractor3_path))

            # clone sketch whole
            sketch_crop_path = gen_crop_name(sketch_path, i)
            shutil.copy(os.path.join(data_dir, 'sketch', sketch_path), 
                        os.path.join(data_dir, 'sketch', sketch_crop_path))

            # if we crop a bad sketch, make sure to record cropped sketches as bad as well
            if sketch_path in bad_games:
                new_bad_games.append(sketch_crop_path.replace('.png', '.png\n'))

            # add a new row in the CSV for each cropped image
            new_row = copy.deepcopy(row)
            new_row[1] = row[1] + '-crop%d' % i
            new_csv_data.append(new_row)

        print('Progress: [{}/{}] files finished augmentation regime.'.format(
            ix+1, len(csv_data)))

    with open(os.path.join(data_dir, 'incorrect_trial_paths_pilot2.txt'), 'a') as fp:
        fp.writelines(new_bad_games)

    print('Wrote new examples to incorrect_trial_paths_pilot2.txt file.')

    with open(os.path.join(data_dir, 'sketchpad_basic_pilot2_group_data.csv'), 'a') as fp:
        writer = csv.writer(fp)
        for row in new_csv_data:
            writer.writerow(row)

    print('Wrote new examples to sketchpad_basic_pilot2_group_data.csv file.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('augment', type=str, help='photo|sketch')
    args = parser.parse_args()
    assert args.augment in ['photo', 'sketch']

    if args.augment == 'sketch':
        augment_by_sketch()
    elif args.augment == 'photo':
        augment_by_photo()
