from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from generators import L2TrainGenerator
from model import L2EmbedNet

import torch
import torch.optim as optim
from torch.autograd import Variable

from utils import pearsonr, corrcoef
from utils import list_files, get_same_photo_sketch_from_photo


def rdm_regularize(model, photo_emb_dir, sketch_emb_dir, n_classes=10, 
                   n_samples=10, diag_lambda=1, xterm_lambda=1, use_cuda=False):
    """Regularize by calculating the euclidean distance between a 
    ground truth RDM on photos and a RDM on sketches across 125 classes
    with n_samples from each class. Notably, this RDM is computed on top
    of embeddings from the multimodal model at training step k.

    NOTE: this is different than the rdm in multimodal_test because this 
    includes both diagonal terms (as in multimodal_test) and cross terms.

    :param model: current multimodal model
    :param photo_emb_dir: directory with photo embeddings with folders 
                          per class
    :param sketch_emb_dir: directory with sketch embeddings with folders
                           per class
    :param n_classes: number of classes across photos/sketches (default: 10)
    :param n_samples: number of samples per sketch/image class (default: 10)
    :param diag_lambda: scaling factor for diag_rdm loss (default: 1)
    :param xterm_lambda: scaling factor for xterm_rdm loss (default: 1)
    :param use_cuda: True if we want to cast cuda tensor types
    """
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    n_samples = max(min(n_samples, 1), 100)
    folders = os.listdir(photo_emb_dir)
    assert len(folders) == 125
    folders = np.random.choice(folders, size=n_classes, replace=False)

    diag_dists = 0.  # keep track of diag distances

    # keep these to store centroids
    photo_centroid_embs = np.zeros((n_classes, 4096))
    sketch_centroid_embs = np.zeros((n_classes, 4096))
    
    # first loop across classes and for each class we will
    # calculate cross & diag RDMs
    for i in range(n_classes):
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

        photo_centroid_embs[i, :] = np.mean(photo_embs, dim=1)
        sketch_centroid_embs[i, :] = np.mean(sketch_embs, dim=1)

        # convert to torch (we will always be in volatile mode)
        photo_embs = Variable(torch.from_numpy(photo_embs).type(dtype))
        sketch_embs = Variable(torch.from_numpy(sketch_embs).type(dtype))

        # probably don't need to do batch-based processing
        photo_embs = model.photo_adaptor(photo_embs)
        sketch_embs = model.sketch_adaptor(sketch_embs)

        # compute cross-correlations
        photo_rdm = corrcoef(photo_embs)
        sketch_rdm = corrcoef(sketch_embs)

        # compute some similarity between these two RDMs (triu)
        rdm_sim = pearsonr(photo_rdm.view(-1), sketch_rdm.view(-1))
        diag_dists += (1 - rdm_sim)  # always positive

    # get average of diag_score
    diag_score = diag_dists / n_classes

    # approximate the cross term embeddings using the centroids for each class
    photo_embs = Variable(torch.from_numpy(photo_centroid_embs).type(dtype))
    sketch_embs = Variable(torch.from_numpy(sketch_centroid_embs).type(dtype))
    photo_embs = model.photo_adaptor(photo_embs)
    sketch_embs = model.sketch_adaptor(sketch_embs)
    photo_rdm = corrcoef(photo_embs)
    sketch_rdm = corrcoef(sketch_embs)
    rdm_sim = pearsonr(photo_rdm.view(-1), sketch_rdm.view(-1))
    xterm_score = 1 - rdm_sim

    return diag_lambda * diag_score + xterm_lambda * xterm_score


if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photo_emb_dir', type=str)
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--rdm_diag_lambda', type=float, default=1.0)
    parser.add_argument('--rdm_xterm_lambda', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    def reset_generators():
        train_generator = L2TrainGenerator(args.photo_emb_dir, args.sketch_emb_dir,
                                           batch_size=args.batch_size, train=True,
                                           use_cuda=args.cuda)
        test_generator = L2TrainGenerator(args.photo_emb_dir, args.sketch_emb_dir,
                                          batch_size=args.batch_size, train=False,
                                          use_cuda=args.cuda)
        return train_generator, test_generator

    train_generator, test_generator = reset_generators()
    train_examples = train_generator.make_generator()
    test_examples = test_generator.make_generator()

    model = L2EmbedNet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    def train(epoch):
        loss_meter = AverageMeter()
        model.train()
        batch_idx = 0
        
        while True:
            try:
                photos, sketches = train_examples.next()
                batch_idx += 1
            except StopIteration:
                break

            optimizer.zero_grad()
            loss = model(photos, sketches)
            loss_meter.update(loss.data[0], photos.size(0)) 

            if args.rdm_lambda > 0:
                model.eval()
                regularization = rdm_regularize(model, args.photo_emb_dir, args.sketch_emb_dir, 
                                                n_classes=5, n_samples=10, diag_lambda=args.rdm_diag_lambda,
                                                xterm_lambda=args.rdm_xterm_lambda, use_cuda=args.cuda)
                # we may be interested in tracking the regularization
                reg_meter.update(regularization.data[0], photos.size(0))
                model.train()
                # loss is the likelihood + regularization term
                loss += regularization
            
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRDM: {:.6f}'.format(
                              epoch, batch_idx * args.batch_size, train_generator.size,
                              (100. * batch_idx * args.batch_size) / train_generator.size,
                              loss_meter.avg, reg_meter.avg), log_path)


    def test(epoch):
        loss_meter = AverageMeter()
        model.eval()
        batch_idx = 0
        
        while True:
            try:
                photos, sketches = test_examples.next()
                batch_idx += 1
            except StopIteration:
                break
            
            loss = model(photos, sketches)
            loss_meter.update(loss.data[0], photos.size(0))

        # no need to track regularization here...
        print_and_log('Test Epoch: {}\tLoss: {:.6f}'.format(
                      epoch, loss_meter.avg), log_path)

        return loss_meter.avg


    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss = test(epoch)

        train_generator, test_generator = reset_generators()
        train_examples = train_generator.make_generator()
        test_examples = test_generator.make_generator()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
            'rdm_lambda': args.rdm_lambda,
        }, is_best, folder=args.out_dir)
