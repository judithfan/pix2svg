"""File for random utility functions"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import torch
from glob import glob


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def merge_sketch_folders(folders, postfixes, out_folder):
    """This will create a new folder with the same structure as the 
    input folders and append things to each folder.

    :param folders: list of strings; each string points to a folder
                    all folders must have the same structure (although
                    they do not need to have the same number of files).
    :param postfixes: list of strings; size must be equal to folders
                      this will add the postfix to every file in that
                      folder in such a way that our embedding generator 
                      will respect it.
    :param out_folder: where to dump the merged content
    """
    clone_folder(folders[0], out_folder)  # all folders should have the same structure
    
    folder_i = 0
    n_folders = len(folders)

    for folder, postfix in zip(folders, postfixes):
        
        filepaths = list_files(folder, ext='png')
        n_files = len(filepaths)
        file_i = 0
        
        for src_path in filepaths:
            src_name, src_ext = os.path.splitext(src_path)
            dest_path = '{name}-{postfix}{ext}'.format(
                name=src_name, postfix=postfix, ext=src_ext)
            dest_path = dest_path.replace(folder, out_folder)
            
            # copy file from src to dest
            shutil.copyfile(src_path, dest_path)
            print('Copied file [{}/{}] of folder [{}/{}]'.format(
                file_i + 1, n_files, folder_i + 1, n_folders))

            file_i += 1

        folder_i += 1


def clone_folder(in_folder, out_folder):
    """Clones all subfolders of in_folder and puts them into
    out_folder. This does not copy files and assumes that 
    out_folder exists already.

    :param in_folder: string
    :param out_folder: string
    """
    all_folders = [x[0] for x in os.walk(in_folder)]
    for folder in all_folders:
        new_folder = folder.replace(in_folder, out_folder)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)


def list_files(path, ext='jpg'):
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.cosine_similarity(input1, input2)
        >>> print(output)
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
