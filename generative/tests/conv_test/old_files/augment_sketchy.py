"""Set up augmentation code for sketchy dataset as well. Perhaps we 
need the visual attention in order to generalize.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
from glob import glob

import augment_sketchpad as augmentlib


# we will only augment by photo since we empirically found that 
# the sketch-augment is not as good.
def augment_by_photo(data_dir):
    sketch_dir = os.path.join(data_dir, 'sketch')
    photo_dir = os.path.join(data_dir, 'photo')

    photo_to_sketch_path = lambda photo_path: \
        photo_path.replace(photo_dir, sketch_dir)

    photo_paths = glob(os.path.join(photo_dir, '*.jpg'))

    for ix, photo_path in enumerate(photo_paths):
        sketch_path = photo_to_sketch_path(photo_path)

        photo = load(photo_path)
        sketch = load(sketch_path)

        photo_crops = augmentlib.obscure(photo, color=127, filter_size=50)
        for i, photo_crop in enumerate(photo_crops):
            photo_im = Image.fromarray(photo_crop)
            photo_crop_path = gen_crop_path(photo_path, i)
            photo_im.save(photo_crop_path)

            # clone sketch (w/o cropping)
            sketch_crop_path = gen_crop_name(sketch_path, i)
            shutil.copy(sketch_path, sketch_crop_path)

        print('Progress: [{}/{}] files finished augmentation regime.'.format(
            ix + 1, len(photo_paths)))


def load(path):
    image = Image.open(path).convert('RGB')
    return np.asarray(image)


def gen_crop_path(path, crop_ix):
    folder = os.path.dirname(path)
    name = os.path.basename(path)
    basename, extension = os.path.splitext(path)
    name = '{base}-crop{ix}{extension}'.format(
        base=basename, ix=crop_ix, extension=extension)
    return os.path.join(folder, name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to folder with data')
    args = parser.parse_args()

    augment_by_photo(args.data_dir)
