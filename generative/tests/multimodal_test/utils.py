"""File for random utility functions"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
from glob import glob


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
