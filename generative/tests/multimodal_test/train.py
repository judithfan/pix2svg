"""A linear/nonlinear transformation from sketch to photo may 
not be enough -- the one-way projection may be too difficult.
We can instead project both the sketch and the photo into a 
shared embeddding space. Then we can randomly sample negatives 
from the same class and from the different class.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import random
import shutil
import numpy as np
from glob import glob

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models
from sklearn.metrics import accuracy_score
from precompute_vgg import list_files

from generators import MultiModalTrainGenerator
from generators import MultiModalTestGenerator

from model import EmbedNet, ConvEmbedNet
from utils import save_checkpoint
from utils import AverageMeter
from utils import EMBED_NET_TYPE, CONV_EMBED_NET_TYPE


def rdm_regularize(model, photo_emb_dir, sketch_emb_dir, n_samples=10, use_cuda=False):
    """Regularize by calculating the euclidean distance between a 
    ground truth RDM on photos and a RDM on sketches across 125 classes
    with n_samples from each class. Notably, this RDM is computed on top
    of embeddings from the multimodal model at training step k.

    :param model: current multimodal model
    :param photo_emb_dir: directory with photo embeddings with folders 
                          per class
    :param sketch_emb_dir: directory with sketch embeddings with folders
                           per class
    :param n_samples: number of samples per sketch/image class (default: 10)
    :param use_cuda: True if we want to cast cuda tensor types
    """
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    n_samples = max(min(n_samples, 1), 100)
    folders = os.listdir(photo_emb_dir)
    assert len(folders) == 125

    dists = 0
    # first loop across classes and for each class we will
    # calculate RDMs
    for i in range(125):
        photo_embs = np.zeros((n_samples, 4096))
        sketch_embs = np.zeros((n_samples, 4096))
        # for each sample, we are going to sample a photo path, and find
        # a corresponding sketch(es)
        photo_paths = list_files(os.path.join(photo_emb_dir, folders[i]), ext='npy')
        photo_paths = np.random.choice(photo_paths, size=n_samples, replace=False)
        for j in range(n_samples):
            sketch_paths = get_same_photo_sketch_from_photo(photo_paths[j], sketch_emb_dir, size=1)
            photo_embs[j, :] = np.load(photo_paths[j])
            sketch_embs[j, :] = np.load(sketch_paths[0])

        # convert to torch (we will always be in volatile mode)
        photo_embs = Variable(torch.from_numpy(photo_embs).type(dtype), volatile=True)
        sketch_embs = Variable(torch.from_numpy(sketch_embs).type(dtype), volatile=True)

        # probably don't need to do batch-based processing
        photo_embs = model.photo_adaptor(photo_embs)
        sketch_embs = model.sketch_adaptor(sketch_embs)

        # compute cross-correlations
        photo_rdm = corrcoef(photo_embs)
        sketch_rdm = corrcoef(sketch_embs)

        # compute some similarity between these two RDMs (triu)
        rdm_sim = pearsonr(photo_rdm.view(-1), sketch_rdm.view(-1))
        dists += (1 - rdm_sim)  # always positive

    return dists


def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor
    
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref: 
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = torch.mm(xm, torch.t(xm))
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


def get_same_photo_sketch_from_photo(photo_path, sketch_emb_dir, size=1):
    """Return n paths to sketches of the provided photo_path

    :param photo_path: string path to photo embedding
    :param sketch_emb_dir: path to folder containing sketch embeddings
    :param size: number to return (default: 1)
    """
    photo_folder = os.path.dirname(photo_path).split('/')[-1]
    photo_filename = os.path.basename(photo_path)
    photo_filename = os.path.splitext(photo_filename)[0]

    # find files in the right sketch folder that start with the photo name
    sketch_paths = glob(os.path.join(sketch_emb_dir, photo_folder, 
                                     '{}-*'.format(photo_filename)))
    sketch_paths = np.random.choice(sketch_paths, size=size)
    return sketch_paths.tolist()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_dir', type=str)
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--convolutional', action='store_true', default=False,
                        help='If True, initialize ConvEmbedNet.')
    parser.add_argument('--rdm_lambda', type=float, default=0.0,
                        help='If >0, add a regularization term to make sketches and photos of the same class closer.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--strict', action='store_true', default=False,
                        help='if True, then consider a sketch of the same class but different photo as negative.')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    def reset_generators():
        train_generator = MultiModalTrainGenerator(args.photo_emb_dir, args.sketch_emb_dir,
                                                   batch_size=args.batch_size,
                                                   strict=args.strict, use_cuda=args.cuda)
        test_generator = MultiModalTestGenerator(args.photo_emb_dir, args.sketch_emb_dir,
                                                 batch_size=args.batch_size, 
                                                 strict=args.strict, use_cuda=args.cuda)
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    train_examples = train_generator.make_generator()
    test_examples = test_generator.make_generator()

    if args.convolutional:
        model = ConvEmbedNet()
        model_type = CONV_EMBED_NET_TYPE
    else:
        model = EmbedNet()
        model_type = EMBED_NET_TYPE

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.cuda:
        model.cuda()


    def train(epoch):
        loss_meter = AverageMeter()
        reg_meter = AverageMeter()
        acc_meter = AverageMeter()
        model.train()
        batch_idx = 0
        
        while True:
            try:
                photos, sketches, labels, _ = train_examples.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            loss_meter.update(loss.data[0], photos.size(0)) 

            if args.rdm_lambda > 0:
                model.eval()
                rdm_dists = rdm_regularize(model, args.photo_emb_dir, args.sketch_emb_dir, 
                                           n_samples=10, use_cuda=args.cuda)
                regularization = args.rdm_lambda * rdm_dists
                # we may be interested in tracking the regularization
                reg_meter.update(regularization.data[0], photos.size(0))
                model.train()
                # loss is the likelihood + regularization term
                loss += regularization
            
            labels_np = labels.cpu().data.numpy()
            outputs_np = np.round(outputs.cpu().squeeze(1).data.numpy(), 0)
            acc = accuracy_score(labels_np, outputs_np)
            acc_meter.update(acc, photos.size(0))
            
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tReg: {:.6f}\tAcc: {:.6f}'.format(
                      epoch, batch_idx * args.batch_size, train_generator.size,
                      (100. * batch_idx * args.batch_size) / train_generator.size,
                      loss_meter.avg, reg_meter.avg, acc_meter.avg))


    def test(epoch):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        model.eval()
        batch_idx = 0
        
        while True:
            try:
                photos, sketches, labels, _ = test_examples.next()
                batch_idx += 1
            except StopIteration:
                break
            
            outputs = model(photos, sketches)
            loss = F.binary_cross_entropy(outputs.squeeze(1), labels.float())
            
            labels_np = labels.cpu().data.numpy()
            outputs_np = np.round(outputs.cpu().squeeze(1).data.numpy(), 0)
            acc = accuracy_score(labels_np, outputs_np)
            acc_meter.update(acc, photos.size(0))
            loss_meter.update(loss.data[0], photos.size(0))

        # no need to track regularization here...
        print('Test Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}'.format(
              epoch, loss_meter.avg, acc_meter.avg))

        return acc_meter.avg


    print('')
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = test(epoch)

        train_generator, test_generator = reset_generators()
        train_examples = train_generator.make_generator()
        test_examples = test_generator.make_generator()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            'type': model_type,
            'strict': args.strict,
            'rdm_lambda': args.rdm_lambda,
        }, is_best, folder=args.out_dir)
