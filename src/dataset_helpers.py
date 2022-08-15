## Standard imports
import numpy as np
import os
import skimage
from skimage.io import imread, imsave, imread
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import filters
from PIL import Image

def reformat(img, type):
    formatted = (img).astype(type)
    return Image.fromarray(formatted)
    
def low_clip(x):
    return np.clip(x, 255, 4095)

def minmax(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def max_normalise(x):
    max = np.max(x)
    return x / max

def is_faulty(x):
    if x.max() <= 255:
        return True
    return False

## DATASET OPERATIONS

RS = 2211

# train_test_split simplified
def unishuffle(a, b, random_state=None):
    assert len(a) == len(b)
    if random_state:
        np.random.seed(RS)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def is_dmso(file):
    # file format: folder/CKX - L - 00(...)

    # only file, no folder
    file = file.split('/')[-1].split("(")[0]

    # get letter for DMSO indices
    letter = file.split('-')[1].strip()

    # number
    label = file[-2:].strip()

    # cell case
    ck = file[:4]

    if ck == "CK19":
        if label in ["5", "8", "11", "15", "18", "21"] and letter in ["N", "O", "P"]:
            return True
    elif ck == "CK21" or ck == "CK22":
        if label not in ["01", "12", "13", "24"] and letter in ["H", "P"]:
            return True
    else:
        print("No CK found")
    return False

def get_label(filename):
    # 0: unstimulated
    # 1: OVA
    # 2: ConA
    # 3: empty

    # filename format: folder/CKX - L - 00(...)
    file = filename.split("/")[-1].split("(")[0]

    # get letter for DMSO indices
    letter = file.split('-')[1].strip()

    # get number
    number = file[-2:].strip()

    # get plate layout number
    ck = file[:4]

    # DMSO = []

    if ck == "CK19":
        if number in ["3", "4", "5", "6", "7", "8", "24"]:
            label = 0
        elif number in ["9", "10", "11", "13", "14", "15", "23"]:
            label = 1
        elif number in ["16", "17", "18", "19", "20", "21", "22"]:
            label = 2
        else:
            label = 3
    elif ck == "CK21" or ck == "CK22":
        if number in ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]:
            label = 0
        elif int(number) in range(14, 24):
            label = 2
        else:
            label = 3
    else:
        print("No CK found")
        return False

    return label

def read_folder_filenames(folder):
    return [os.path.join(folder, f)for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f[0] != '.' and "Brightfield" not in f]


def sliding_window(img, dest_size, rgb=False):
    new_img = np.full_like(img, img)

    size = img.shape[0]
    if dest_size > size or dest_size % 2 != 0:
        raise Exception("destination size is bigger than picture size or destination size is not even")

    qty = size//dest_size
    if size % dest_size != 0:
        # need to crop out the left and bottom (less significant in dataset)
        crop = size-dest_size*qty
        new_img = new_img[crop:, :-crop]

    if rgb:
        windows = np.ndarray(shape=(qty**2, dest_size, dest_size, 3), dtype=np.uint16)
    else:
        windows = np.ndarray(shape=(qty**2, dest_size, dest_size), dtype=np.uint16)

    i = 0
    for row in range(qty):
        y = row*dest_size
        x = 0
        for col in range(qty):
            #print("x:coord {},{} - y:coord {},{}".format(x, x+dest_size, y, y+dest_size))
            windows[i] = new_img[x:x+dest_size, y:y+dest_size]
            x += dest_size
            i += 1

    return windows

# reverses sliding window
def reconstruct_from(images, size=192):
    # work on reconstruction
    new_img = np.ndarray(shape=(size*10, size*10), dtype=np.float32)

    col = 0
    y = 0
    x = 0
    for i in range(100):
        new_img[x:x+size, y:y+size] = images[i]
        x = size*col
        col += 1
        if col == 10:
            col = 0
            y += size

    return new_img
