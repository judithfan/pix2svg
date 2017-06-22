import numpy as np
from numpy import *
import os
import sys
import caffe


OUTPUT_DIR = './triplet_features'
sketchy_dir = './sketchy_triplet_network'
PRETRAINED_FILE = os.path.join(sketchy_dir,'triplet_googlenet_finegrain_final.caffemodel')
sketch_model = os.path.join(sketchy_dir,'Triplet_googlenet_sketchdeploy.prototxt')
image_model = os.path.join(sketchy_dir,'Triplet_googlenet_imagedeploy.prototxt')

caffe.set_mode_gpu()
#caffe.set_mode_cpu()
sketch_net = caffe.Net(sketch_model, PRETRAINED_FILE, caffe.TEST)
img_net = caffe.Net(image_model, PRETRAINED_FILE, caffe.TEST)
sketch_net.blobs.keys()

# set output layer name
output_layer_sketch = 'pool5/7x7_s1_s'
output_layer_image = 'pool5/7x7_s1_p'


#set the transformer
transformer = caffe.io.Transformer({'data': np.shape(sketch_net.blobs['data'].data)})
transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_transpose('data',(2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

# sketchy sketch/photo dataset
categories = os.listdir('./sketchy_dataset/photos/')

photo_feats = []
sketch_feats = []
photo_fnames = []
sketch_fnames = []

for c in categories:    
    photo_path = os.path.join('./sketchy_dataset/photos',c)
    sketch_path = os.path.join('./sketchy_dataset/sketches',c)
    photo_list = os.listdir(photo_path)
    sketch_list = os.listdir(sketch_path)
    for p in photo_list:
        print('Extracting: ', p)
        imgname = p
        full_path = os.path.join(photo_path,p)
        photo_fnames.append(full_path)
        img = transformer.preprocess('data', caffe.io.load_image(full_path))
        img_in = np.reshape([img],np.shape(img_net.blobs['data'].data))
        out_img = img_net.forward(data=img_in)
        out_img = np.copy(out_img[output_layer_image])
        photo_feats.append(out_img)
    for s in sketch_list:
        print('Extracting: ', s)
        full_path = os.path.join(sketch_path,s)
        sketch_fnames.append(full_path)
        img = transformer.preprocess('data', caffe.io.load_image(full_path))
        img_in = np.reshape([img],np.shape(sketch_net.blobs['data'].data))
        out_img = sketch_net.forward(data=img_in)
        out_img = np.copy(out_img[output_layer_sketch])
        sketch_feats.append(out_img)
        
print('Done!')

photo_feats = np.array(np.resize(photo_feats,[np.shape(photo_feats)[0],np.shape(photo_feats)[2]]))
sketch_feats = np.array(np.resize(sketch_feats,[np.shape(sketch_feats)[0],np.shape(sketch_feats)[2]]))

photo_fnames,sketch_fnames = map(np.array,[photo_fnames,sketch_fnames])

np.save(os.path.join(OUTPUT_DIR,'photo_features.npy'), photo_feats)
np.save(os.path.join(OUTPUT_DIR,'sketch_features.npy'), sketch_feats)
                                        
np.savetxt(os.path.join(OUTPUT_DIR,'photo_filenames.txt'), photo_fnames, fmt="%s")
np.savetxt(os.path.join(OUTPUT_DIR,'sketch_filenames.txt'), sketch_fnames, fmt="%s")
