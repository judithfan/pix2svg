from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import copy
from glob import glob
import numpy as np
from PIL import Image
from copy import deepcopy

from torch.autograd import Variable
import torchvision.transforms as transforms


def list_files(path, ext='jpg'):
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result


def load_image(path, imsize=256, volatile=True, use_cuda=False):
    im = Image.open(path)
    im = im.convert('RGB')

    loader = transforms.Compose([
        transforms.Scale(imsize),
        transforms.ToTensor()])

    im = Variable(loader(im), volatile=volatile)
    im = im.unsqueeze(0)
    if use_cuda:
        im = im.cuda()
    return im


def train_test_generator(imsize=256, train=True, use_cuda=False):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'

    categories = os.listdir(sketch_dir)
    n_categories = len(categories)
    if train:
        categories = categories[:int(n_categories * 0.8)]
    else:
        categories = categories[int(n_categories * 0.8):]

    photo_paths = [path for path in list_files(photo_dir, ext='jpg') 
                   if os.path.dirname(path).split('/')[-1] in categories]
    sketch_paths = [path for path in list_files(sketch_dir, ext='png') 
                   if os.path.dirname(path).split('/')[-1] in categories]

    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]

        photo_filename = sketch_filename.split('-')[0] + '.jpg'
        photo_path = os.path.join(photo_dir, sketch_folder, photo_filename)

        photo = load_image(photo_path, imsize=imsize, use_cuda=use_cuda)
        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)

        yield (photo, sketch)


def train_test_size():
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'
    categories = os.listdir(sketch_dir)
    n_categories = len(categories)
    train_categories = categories[:int(n_categories * 0.8)]
    test_categories = categories[int(n_categories * 0.8):]
    train_paths = [path for path in list_files(sketch_dir, ext='png')
                   if os.path.dirname(path).split('/')[-1] in train_categories]
    test_paths = [path for path in list_files(sketch_dir, ext='png')
                  if os.path.dirname(path).split('/')[-1] in test_categories]
    return len(train_paths), len(test_paths)


def data_generator(imsize=256, use_cuda=False, ignore_class=None):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'

    photo_paths = list_files(photo_dir, ext='jpg')
    sketch_paths = list_files(sketch_dir, ext='png')

    if ignore_class:
        photo_paths = [i for i in photo_paths if os.path.dirname(i).split('/')[-1] != ignore_class]
        sketch_paths = [i for i in sketch_paths if os.path.dirname(i).split('/')[-1] != ignore_class]

    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]

        photo_filename = sketch_filename.split('-')[0] + '.jpg'
        photo_path = os.path.join(photo_dir, sketch_folder, photo_filename)

        photo = load_image(photo_path, imsize=imsize, use_cuda=use_cuda)
        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)

        yield (photo, sketch)


def noisy_generator(imsize=256, use_cuda=False, ignore_class=None):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    sketch_dir = '/home/jefan/full_sketchy_dataset/noise'

    photo_paths = list_files(photo_dir, ext='jpg')
    sketch_paths = list_files(sketch_dir, ext='png')

    if ignore_class:
        photo_paths = [i for i in photo_paths if os.path.dirname(i).split('/')[-1] != ignore_class]
        sketch_paths = [i for i in sketch_paths if os.path.dirname(i).split('/')[-1] != ignore_class]

    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]

        photo_filename = sketch_filename.split('-')[0] + '.jpg'
        photo_path = os.path.join(photo_dir, sketch_folder, photo_filename)

        photo = load_image(photo_path, imsize=imsize, use_cuda=use_cuda)
        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)

        yield (photo, sketch)


def swapped_generator(imsize=256, use_cuda=False, ignore_class=None):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'

    photo_paths = list_files(photo_dir, ext='jpg')
    sketch_paths = list_files(sketch_dir, ext='png')
    
    if ignore_class:
        photo_paths = [i for i in photo_paths if os.path.dirname(i).split('/')[-1] != ignore_class]
        sketch_paths = [i for i in sketch_paths if os.path.dirname(i).split('/')[-1] != ignore_class]
    
    # for each sketch, i'm going to randomly sample a photo that is not 
    # in the same parent directory (class).

    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]

        while True:
            random_photo_path = np.random.choice(photo_paths)
            random_photo_folder = os.path.dirname(random_photo_path).split('/')[-1]
            if random_photo_folder != sketch_folder:
                break

        photo = load_image(random_photo_path, imsize=imsize, use_cuda=use_cuda)
        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)

        yield (photo, sketch)


def neighbor_generator(imsize=256, use_cuda=False):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'

#     photo_paths = list_files(photo_dir, ext='jpg')
    sketch_paths = list_files(sketch_dir, ext='png')
    
    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]        
    
        ## sample a different photo from the same class 
        matching_photo = sketch_filename.split('-')[0] + '.jpg'
        matching_photo_path = os.path.join(photo_dir,sketch_folder,matching_photo)   
        photo_class = os.path.join(photo_dir,sketch_folder)            
        while True:                        
            _random_photo_path = np.random.choice(os.listdir(photo_class))
            random_photo_path = os.path.join(photo_class,_random_photo_path)
            if _random_photo_path != sketch_filename:
                break

        photo = load_image(random_photo_path, imsize=imsize, use_cuda=use_cuda)
        sketch = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)

        yield (photo, sketch)        


def sketchspsc_generator(imsize=256, use_cuda=False):
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'
    sketch_paths = list_files(sketch_dir, ext='png') 
    ## This yields, for each sketch, a random other sketch of the SAME PHOTO.
    ## "SPSC" = "same photo, same class"
    
    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]   
        
        while True: 
            other_sketches = [i for i in os.listdir(os.path.join(sketch_dir,sketch_folder)) if sketch_filename.split('-')[0] in i]
            random_other_sketch = np.random.choice(other_sketches)
            random_other_sketch_path = os.path.join(sketch_dir,sketch_folder,random_other_sketch)
            if random_other_sketch != sketch_filename:
                break 
        sketch1 = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)
        sketch2 = load_image(random_other_sketch_path, imsize=imsize, use_cuda=use_cuda)
        
        yield (sketch1, sketch2)                        


def sketchdpsc_generator(imsize=256, use_cuda=False):
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'
    sketch_paths = list_files(sketch_dir, ext='png') 
    
    ## This yields, for each sketch, a random other sketch of a photo from the same CLASS.
    ## "DPSC" = "different photo, same class"
    
    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]   
        
        while True: 
            other_sketches = [i for i in os.listdir(os.path.join(sketch_dir,sketch_folder)) if sketch_filename.split('-')[0] not in i]
            random_other_sketch = np.random.choice(other_sketches)
            random_other_sketch_path = os.path.join(sketch_dir,sketch_folder,random_other_sketch)
            if random_other_sketch != sketch_filename:
                break 
        sketch1 = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)
        sketch2 = load_image(random_other_sketch_path, imsize=imsize, use_cuda=use_cuda)
        
        yield (sketch1, sketch2)


def sketchdpdc_generator(imsize=256, use_cuda=False):
    sketch_dir = '/home/jefan/full_sketchy_dataset/sketches'
    sketch_paths = list_files(sketch_dir, ext='png') 
    ## This yields, for each sketch, a random other sketch of a DIFFERENT photo from a DIFFERENT CLASS.
    ## "DPDC" = "different photo, different class"
    
    for i in range(len(sketch_paths)):
        sketch_path = sketch_paths[i]
        sketch_filename = os.path.basename(sketch_path)
        sketch_folder = os.path.dirname(sketch_path).split('/')[-1]   
        all_classes = np.unique([os.path.dirname(sp).split('/')[-1] for sp in sketch_paths])
        other_classes = [i for i in all_classes if i not in sketch_folder]
        
        # select random other class, then random sketch within that class
        random_other_class = np.random.choice(other_classes)
        other_sketches = os.listdir(os.path.join(sketch_dir,random_other_class))
        random_other_sketch = np.random.choice(other_sketches)
        random_other_sketch_path = os.path.join(sketch_dir,random_other_class,random_other_sketch)
  
        sketch1 = load_image(sketch_path, imsize=imsize, use_cuda=use_cuda)
        sketch2 = load_image(random_other_sketch_path, imsize=imsize, use_cuda=use_cuda)
        
        yield (sketch1, sketch2)                        


def photodpsc_generator(imsize=256, use_cuda=False):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    photo_paths = list_files(photo_dir, ext='jpg') 
    ## This yields, for each photo, a random other photo from the same class.
    ## "DPSC" = "different photo, same class"
    
    for i in range(len(photo_paths)):
        photo_path = photo_paths[i]
        photo_filename = os.path.basename(photo_path)
        photo_folder = os.path.dirname(photo_path).split('/')[-1]   
        
        # get list of remaining photos in this directory
        other_photos = [i for i in os.listdir(os.path.join(photo_dir,photo_folder)) if i != photo_filename]
        random_other_photo = np.random.choice(other_photos)
        random_other_photo_path = os.path.join(photo_dir,photo_folder,random_other_photo)
            
        photo1 = load_image(photo_path, imsize=imsize, use_cuda=use_cuda)
        photo2 = load_image(random_other_photo_path, imsize=imsize, use_cuda=use_cuda)
        
        yield (photo1, photo2)   


def photodpdc_generator(imsize=256, use_cuda=False):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    photo_paths = list_files(photo_dir, ext='jpg') 
    ## This yields, for each photo, a photo from a random other class.
    ## "DPDC" = "different photo, different class"
    
    all_classes = np.unique([os.path.dirname(sp).split('/')[-1] for sp in photo_paths])    
    for i in range(len(photo_paths)):
        photo_path = photo_paths[i]
        photo_filename = os.path.basename(photo_path)
        photo_folder = os.path.dirname(photo_path).split('/')[-1]   
        other_classes = [i for i in all_classes if i != photo_folder]  
        assert len(other_classes)==len(all_classes)-1
        
        # get list of remaining photos in this directory
        random_other_class = np.random.choice(other_classes)
        other_photos = os.listdir(os.path.join(photo_dir,random_other_class))
        random_other_photo = np.random.choice(other_photos)
        random_other_photo_path = os.path.join(photo_dir,random_other_class,random_other_photo)
            
        photo1 = load_image(photo_path, imsize=imsize, use_cuda=use_cuda)
        photo2 = load_image(random_other_photo_path, imsize=imsize, use_cuda=use_cuda)
        
        yield (photo1, photo2)         


def perturbed_generator(imsize=256, use_cuda=False, n_perturbations_per_image=5):
    photo_dir = '/home/jefan/full_sketchy_dataset/photos'
    photo_paths = list_files(photo_dir, ext='jpg')

    for i in range(len(photo_paths)):
        photo_path = photo_paths[i]
        photo = load_image(photo_path, imsize=imsize, use_cuda=use_cuda)
        
        for j in range(n_perturbations_per_image):
            perturbed_photo = add_gaussian_noise(photo, imsize=imsize, std=0.1, use_cuda=use_cuda)
            yield (photo, perturbed_photo)


def add_salt_and_pepper(image, imsize=224, amount=0.01):
    im = copy.deepcopy(image)
    
    num_salt = int(np.ceil(amount * np.prod(im.size()) * s_vs_p))
    x_noise = np.random.randint(0, imsize, num_salt)
    y_noise = np.random.randint(0, imsize, num_salt)
    for x, y in zip(x_noise, y_noise):
        im[:, x, y] = 0

    num_pepper = int(np.ceil(amount* np.prod(im.size()) * (1. - s_vs_p)))
    x_noise = np.random.randint(0, imsize, num_pepper)
    y_noise = np.random.randint(0, imsize, num_pepper)
    
    for x, y in zip(x_noise, y_noise):
        im[:, x, y] = 1

    return im


def add_gaussian_noise(image, imsize=224, std=0.1, use_cuda=False):
    im = copy.deepcopy(image) 
    noise = torch.normal(0, torch.ones(imsize * imsize * 3) * std).view((3, imsize, imsize))
    noise = noise.unsqueeze(0)
    noise = Variable(noise)
    if use_cuda:
        noise = noise.cuda()
    im = im + noise
    im = im.clamp(0, 1)
    return im

