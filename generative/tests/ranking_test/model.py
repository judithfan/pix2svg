"""Similar to multimodal model but uses L2 distance with a 
ranking loss similar to search.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SketchRankNet(nn.Module):
    def __init__(self):
        super(SketchRankNet, self).__init__()
        self.photo_adaptor = AdaptorNet(4096, 1000)
        self.sketch_adaptor = AdaptorNet(4096, 1000)
        self.fusenet = FuseEuclidean()

    def forward(self, photo_emb, sketch_same_photo_emb, 
                sketch_same_class_emb, sketch_diff_class_emb, noise_emb):
        photo_emb = self.photo_adaptor(photo_emb)
        sketch_same_photo_emb = self.sketch_adaptor(sketch_same_photo_emb)
        sketch_same_class_emb = self.sketch_adaptor(sketch_same_class_emb)
        sketch_diff_class_emb = self.sketch_adaptor(sketch_diff_class_emb)
        noise_emb = self.sketch_adaptor(noise_emb)

        # compute distances between photo and each sketch
        photo_sketch_same_photo_dist = self.fusenet(photo_emb, sketch_same_photo_emb)
        photo_sketch_same_class_dist = self.fusenet(photo_emb, sketch_same_class_emb)
        photo_sketch_diff_class_dist = self.fusenet(photo_emb, sketch_diff_class_emb)
        photo_noise_dist = self.fusenet(photo_emb, noise_emb)

        # combine dists into a tensor
        dists = torch.cat([photo_sketch_same_photo_dist, 
                           photo_sketch_same_class_dist,
                           photo_sketch_diff_class_dist, 
                           photo_noise_dist], dim=1)

        return dists


def ranking_loss(dists, strong=False, use_cuda=False):
    # https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    p = dists.size()[1]
    discount = torch.from_numpy(np.log2(np.arange(1, p + 1) + 1)).type(dtype)
    discount = discount.unsqueeze(0).repeat(dists.size()[0], 1)
    discount = Variable(discount)
    if strong:
        base2 = torch.Tensor([2]).repeat(p).type(dtype)
        dists = torch.pow(base2, dists) - 1
    dists = torch.sum(dists / discount, dim=1)
    return torch.mean(dists)


class FuseEuclidean(nn.Module):
    def forward(self, e1, e2):
        # center cosine similarity (pearson coefficient)
        e1 = e1 - torch.mean(e1, dim=1, keepdim=True)
        e2 = e2 - torch.mean(e2, dim=1, keepdim=True)
        return torch.norm(e1 - e2, dim=1, p=2, keepdim=True)


class AdaptorNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AdaptorNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, (in_dim + out_dim) // 2)
        self.fc2 = nn.Linear((in_dim + out_dim) // 2, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.drop1(x)
        x = F.elu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


def save_checkpoint(state, is_best, folder='./',
                    filename='checkpoint.pth.tar'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)

    model = SketchRankNet()
    model.load_state_dict(checkpoint['state_dict'])
    return model

