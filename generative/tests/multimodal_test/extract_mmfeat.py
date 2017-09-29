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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms

from multimodaltest import EmbedNet
import sys
sys.path.append('../distribution_test')
from distribtest import cosine_similarity

def load_checkpoint(file_path, use_cuda=False):
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)

    model = EmbedNet(checkpoint['adaptive_size'])  #import me
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_num_paths(base_path):
    sketch_dir = os.path.join(base_path,'sketch')  
    feat_list = os.listdir(sketch_dir)
    return len(feat_list)

def pair_generator(base_path, referent, batch_size=32, use_cuda=True):    
    sketch_dir = os.path.join(base_path,'sketch')    
    target_dir = os.path.join(base_path,'target')
    distractor1_dir = os.path.join(base_path,'distractor1')
    distractor2_dir = os.path.join(base_path,'distractor2')
    distractor3_dir = os.path.join(base_path,'distractor3')   
    
    feat_list = os.listdir(sketch_dir)
    sketch_paths = [os.path.join(base_path,sketch_dir,i) for i in feat_list]
    if referent=='target':
        referent_paths = [os.path.join(base_path,target_dir,i) for i in feat_list]
    elif referent=='distractor1_dir':
        referent_paths = [os.path.join(base_path,distractor1_dir,i) for i in feat_list]
    elif referent=='distractor2_dir':
        referent_paths = [os.path.join(base_path,distractor2_dir,i) for i in feat_list]
    elif referent=='distractor3_dir':
        referent_paths = [os.path.join(base_path,distractor3_dir,i) for i in feat_list] 
        
    zipped = zip(sketch_paths,referent_paths)        
        
    for path in zipped:
        sketch_feat = np.load(path[0])
        referent_feat = np.load(path[1])
        yield(sketch_feat, referent_feat)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default = '/home/jefan/multimodal_embedding/9_28_17/trained_model_nonstrict_2/model_best.pth.tar')
    parser.add_argument('--base_path', type=str, default='/home/jefan/sketchpad_basic_features')  
    parser.add_argument('--out_dir', type=str, default='/home/jefan/sketchpad_basic_multimodal_features')      
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda_device', type=int, default=0)
    args = parser.parse_args()
    
    # load in model & freeze
    model = load_checkpoint(args.model_path, use_cuda=True)
    referent_adaptor = model.photo_adaptor
    sketch_adaptor = model.sketch_adaptor        
    referent_adaptor.eval()
    sketch_adaptor.eval()
    
    use_cuda = torch.cuda.is_available()
    cuda_device = args.cuda_device    

    # define generator
    referent = 'target'
    generator = pair_generator(args.base_path, referent, batch_size=args.batch_size, use_cuda=True)

    n = 0
    quit = False 
    num_sketches = get_num_paths(args.base_path)
    
    Referent_feat = []
    Sketch_feat = []   
    SR_similarity = []
    
    if generator:
        while True:    
            batch_size = args.batch_size
            sketch_batch = Variable(torch.zeros(args.batch_size, 4096))
            referent_batch = Variable(torch.zeros(args.batch_size, 4096))
            if use_cuda:
                sketch_batch = sketch_batch.cuda(cuda_device)   
            print('Batch {}'.format(n + 1))            
            for b in range(args.batch_size):                
                try:
                    sketch, referent = generator.next()
                    sketch_batch[b] = torch.from_numpy(sketch).cuda(args.cuda_device)
                    referent_batch[b] = torch.from_numpy(referent).cuda(args.cuda_device)
                except StopIteration:
                    quit = True
                    print('stopped!')
                    break                     
            n += 1                         
            print(sketch_batch)
            referent_feat = referent_adaptor(referent_batch)  # batch, 4096
            sketch_feat = sketch_adaptor(sketch_batch)  # batch, 4096
            
            if len(Referent_feat)==0:
                Referent_feat = referent_feat
                Sketch_feat = sketch_feat
            else:
                Referent_feat = np.vstack((Referent_feat,referent_feat))      
                Sketch_feat = np.vstack((Sketch_feat,sketch_feat)) 
                
            SR_similarity.append(cosine_similarity(photo_emb, sketch_emb, dim=1).unsqueeze(1))
                
            if n == num_sketches//args.batch_size + 1:
                break                

    SR_similarity = np.array([item for sublist in SR_similarity for item in sublist])
    
    Referent_out_path = os.path.join(args.out_dir,referent+'.npy')
    Sketch_out_path = os.path.join(args.out_dir,'sketch.npy')   
    SR_similarity_out_path =  os.path.join(args.out_dir,'sketch_{}_similarity.npy'.format(referent))
    
    print('Saving out .npy with embeddings and distances between sketch and {}'.format(referent))
    np.save(Referent_out_path,Referent_feat)
    np.save(Sketch_out_path,Sketch_feat)
    np.save(SR_similarity_out_path,SR_simliarity)    
    
