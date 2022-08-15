from dataset_helpers import read_folder_filenames, is_dmso
from dataset_helpers import sliding_window, get_label

from skimage.io import imread
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser(description='Compress images into efficient Numpy NPZ file.\n Images are split up into 192x192')

parser.add_argument('--folder', '-f', action='store', type=str, help='Folder to parse files from', required=True)
parser.add_argument('--out', '-o', action='store', type=str, help='Filepath to compress to, will send to current dir if unspecified')

args = parser.parse_args()
folder = args.folder
out = args.out

# read all filenames
filenames = sorted(read_folder_filenames(folder))

if not out:
    out = "compressed.npz"
print("Compressing from {}\ninto {}".format(folder, out))

# compress images
def compress_images(out, filenames, size):
    """
    returns:
    a npz file of:
     - image arrays in shape (size, size, 1)
     - filenames (unmodified)

    @parameters:
    out = name of the outputted compressed file
    filenames = all filenames of files to compress
    size = size of output images


    @assumptions:
    * validity of filenames has been checked
    """

    compressed = []
    fn = []

    for file in filenames:
        img = imread(file)
        windows = sliding_window(img, size)
        img = None
        for img in windows:
            compressed.append(img)
            fn.append(file)
            img = None
        windows = None

    compressed = np.array(compressed)
    fn = np.array(fn)
    np.savez(out, x=compressed, y=fn)

    print("All files compressed into %s" % out)

compress_images(out, filenames, 192)
