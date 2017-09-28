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


class SketchDataset():
    """dataset of photos and sketches for pix2svg."""

    def __init__(self, npy_file, photo_dir, class_name, transform=None):
        """
        Args:
            npy_file (string): Path to the numpy file with stroke-5 representation and corresponding photos.
                    # to get stroke-5 representation of svg
                    x['airplane'][0][5]
                    # to get corresponding sketch path
                    x['airplane'][1][5]
            photo_dir (string): Directory with all the photos.
            class_name: name of category (e.g., airplane)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.stroke_dir = npy_file
        self.photo_dir = photo_dir
        self.class_name = class_name
        self.strokes = np.load(npy_file)[()]
        self.transform = transform
    
    def __len__(self):
        return len(self.strokes[self.class_name][0])

    def __getitem__(self, idx):
        X = self.strokes
        filelist = X[self.class_name][1]
        photo_filename = filelist[idx].split('/')[-1].split('.')[0].split('-')[0] + '.jpg'
        if self.class_name=='car_(sedan)':
            _cname = 'car'
        else:
            _cname = self.class_name        
        photo_path = os.path.join(self.photo_dir,_cname,photo_filename)                               
        photo = io.imread(photo_path)
        photo = photo.astype(float)
        strokes = self.strokes[self.class_name][0][idx]
        sketch_filename = filelist[idx].split('/')[-1].split('.')[0] + '.png'
        sample = {'photo': photo, 'strokes': strokes,'name': photo_filename, 'sketch_filename': sketch_filename}          
        
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

        
def flatten(x):
    return [item for sublist in x for item in sublist]        
        
if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./sketch_coords')    
    parser.add_argument('--save_dir', type=str, default='./stroke_dataframes')   
    parser.add_argument('--photo_dir', type=str, default='/home/jefan/full_sketchy_dataset/photos')
    
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    all_classes = os.listdir(args.root_dir)
    
    for c in all_classes:
        cname = c.split('.')[0][7:]
        
        ## load in each dataset
        Class = SketchDataset(npy_file=os.path.join(args.root_dir,c), \
                                photo_dir=args.photo_dir, class_name=cname, transform=None)    

        ## loop through Class and save photos and sketchID's indexed in same way as full_sketchy_dataset is organized
        photos = []
        sketchID = []
        sketchFN = []
        counter = 1
        for i in range(len(Class)):
            photos.append(Class[i]['name'].split('.')[0])
            sketchFN.append(Class[i]['sketch_filename'])
            if i==0:
                sketchID.append(1)
            elif Class[i]['name'].split('.')[0] == Class[i-1]['name'].split('.')[0]: # current matches previous
                counter = counter + 1
                sketchID.append(counter)
            elif Class[i]['name'].split('.')[0] != Class[i-1]['name'].split('.')[0]: # new photo dir
                counter = 1
                sketchID.append(counter)

        unique_photos = np.unique(photos)
        zipped = zip(photos,sketchID,sketchFN)    

        ##### save out full stroke matrix (55855,5): columns are: [x,y,pen_state,sketch_id,photo_id]
        Verts = []
        Codes = []
        PhotoID = [] # object-level
        SketchID = [] # sketch-level
        StrokeID = [] # stroke-level
        SketchFN = [] # sketch png filename

        for idx in range(len(Class)):
            sample = Class[idx]
            this_sketchFN = zipped[idx][2]
            this_sketchID = zipped[idx][1]
            this_photoID = zipped[idx][0]
            lines = strokes_to_lines(to_normal_strokes(sample['strokes']))
            verts,codes = polyline_pathmaker(lines)
            Verts.append(verts)
            Codes.append(codes)
            SketchID.append([this_sketchID]*len(verts))
            PhotoID.append([this_photoID]*len(verts))
            SketchFN.append([this_sketchFN]*len(verts))
            strokeID = []
            for i,l in enumerate(lines):
                strokeID.append([i]*len(l))
            StrokeID.append(flatten(strokeID))


        Verts,Codes,SketchID,PhotoID,StrokeID,SketchFN = map(flatten,[Verts,Codes,SketchID,PhotoID,StrokeID,SketchFN]) 
        x,y = zip(*Verts) # unzip x,y segments 
        print str(len(Verts)) + ' vertices to predict.'

        data = np.array([x,y,Codes,StrokeID,SketchID,PhotoID, SketchFN]).T
        S = pd.DataFrame(data,columns=['x','y','pen','strokeID','sketchID','photoID','sketchFN'])
        print 'Saving out stroke_dataframe for ' + cname
        save_path = os.path.join(args.save_dir,cname + '.csv')
        S.to_csv(save_path) 
    
    