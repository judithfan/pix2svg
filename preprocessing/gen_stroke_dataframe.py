from __future__ import division

import numpy as np
from numpy import *

import os

import PIL
from PIL import Image
import matplotlib.pyplot as plt

from skimage import data, io, filters

from matplotlib.path import Path
import matplotlib.patches as patches

import pandas as pd
import helpers as helpers


class TinyDataset():
    """tiny airplane dataset of photos and sketches for pix2svg."""

    def __init__(self, npy_file, root_dir, transform=None):
        """
        Args:
            npy_file (string): Path to the numpy file with stroke-5 representation and corresponding photos.
                    # to get stroke-5 representation of svg
                    x['airplane'][0][5]
                    # to get corresponding photos
                    x['airplane'][1][5]
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.stroke_dir = npy_file
        self.photo_dir = os.path.join(root_dir,'photo')
        self.stroke_numpy = np.load(os.path.join(root_dir,'airplane.npy'))[()]
        self.strokes = np.load(npy_file)[()]
        self.transform = transform
    
    def __len__(self):
        return len(self.strokes['airplane'][0])

    def __getitem__(self, idx):
        X = self.stroke_numpy
        img_name = os.path.join(self.photo_dir,'airplane',X['airplane'][1][idx]+ '.jpg')
        photo = io.imread(img_name)
        photo = photo.astype(float)
        strokes = self.strokes['airplane'][0][idx]
        sample = {'photo': photo, 'strokes': strokes,'name': X['airplane'][1][idx]+ '.jpg'}
        
        if self.transform:
            sample = self.transform(sample)

        return sample    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, strokes, name = sample['photo'], sample['strokes'], sample['name']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))        
        return {'tensor': tf.divide(tf.stack(sample['photo']),255),
                'strokes': strokes,
                'name': name,
                'photo': image}    

def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result    
    
def strokes_to_lines(strokes):
    """
    Convert stroke-3 format to polyline format.
    List contains sublist of continuous line segments (strokes).    
    """
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines

def polyline_pathmaker(lines):
    x = []
    y = []

    codes = [Path.MOVETO] # start with moveto command always
    for i,l in enumerate(lines):
        for _i,_l in enumerate(l):
            x.append(_l[0])
            y.append(_l[1])
            if _i<len(l)-1:
                codes.append(Path.LINETO) # keep pen on page
            else:
                if i != len(lines)-1: # final vertex
                    codes.append(Path.MOVETO)
    verts = zip(x,y)            
    return verts, codes

def path_renderer(verts, codes):
    if len(verts)>0:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)
        ax.set_xlim(0,640)
        ax.set_ylim(0,640) 
        ax.axis('off')
        plt.gca().invert_yaxis() # y values increase as you go down in image
        plt.show()
    else:
        ax.set_xlim(0,640)
        ax.set_ylim(0,640)        
        ax.axis('off')
        plt.show()

        
if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', type=str, default='./tiny/airplane.npy')
    parser.add_argument('--root_dir', type=str, default='./tiny')    
    parser.add_argument('--save_path', type=str, default='./tiny/stroke_dataframe.csv')    
    
    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    
    ## load in airplanes dataset
    airplanes = TinyDataset(npy_file=args.npy_path, \
                            root_dir=args.root_dir,transform=None)    

    ## loop through airplanes and save photos and sketchID's indexed in same way as full_sketchy_dataset is organized
    photos = []
    sketchID = []
    counter = 1
    for i in range(len(airplanes)):
        photos.append(airplanes[i]['name'].split('.')[0])
        if i==0:
            sketchID.append(1)
        elif airplanes[i]['name'].split('.')[0] == airplanes[i-1]['name'].split('.')[0]: # current matches previous
            counter = counter + 1
            sketchID.append(counter)
        elif airplanes[i]['name'].split('.')[0] != airplanes[i-1]['name'].split('.')[0]: # new photo dir
            counter = 1
            sketchID.append(counter)

    unique_photos = np.unique(photos)
    zipped = zip(photos,sketchID)    
    
    ##### save out full stroke matrix (55855,5): columns are: [x,y,pen_state,sketch_id,photo_id]
    Verts = []
    Codes = []
    PhotoID = [] # object-level
    SketchID = [] # sketch-level
    StrokeID = [] # stroke-level

    for idx in range(len(airplanes)):
        sample = airplanes[idx]
        this_sketchID = zipped[idx][1]
        this_photoID = zipped[idx][0]
        lines = strokes_to_lines(to_normal_strokes(sample['strokes']))
        verts,codes = polyline_pathmaker(lines)
        Verts.append(verts)
        Codes.append(codes)
        SketchID.append([this_sketchID]*len(verts))
        PhotoID.append([this_photoID]*len(verts))
        strokeID = []
        for i,l in enumerate(lines):
            strokeID.append([i]*len(l))
        StrokeID.append(flatten(strokeID))

    def flatten(x):
        return [item for sublist in x for item in sublist]
    Verts,Codes,SketchID,PhotoID,StrokeID = map(flatten,[Verts,Codes,SketchID,PhotoID,StrokeID]) 
    x,y = zip(*Verts) # unzip x,y segments 
    print str(len(Verts)) + ' vertices to predict.'

    data = np.array([x,y,Codes,StrokeID, SketchID,PhotoID]).T
    S = pd.DataFrame(data,columns=['x','y','pen','strokeID','sketchID','photoID'])
    print 'Saving out stroke_dataframe.csv'
    S.to_csv(args.save_path)    
    
    
    
    