## Standard imports
import numpy as np
import os
import skimage
from skimage.io import imread, imsave, imread
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import filters
from PIL import Image

# IMAGE FORMATTING OPERATIONS

imw, imh, c = 192, 192, 1

def reshape(img, w=imw, h=imh, c=c):
    if c > 1:
      return np.reshape(img, (w, h, c))
    else:
      return np.reshape(img, (w, h))

def reformat(img, type):
    formatted = (img).astype(type)
    return Image.fromarray(formatted)

def center_crop(img, size=imw):
    to_crop = (img.shape[0]-size)/2
    image_resized = skimage.util.crop(img, (to_crop, to_crop))
    return image_resized

def normalise(img):
    # normalise 16-bit TIF image
    img = np.full_like(img, img)
    return img / 65535.0

def mean_clip(x):
    mean = np.mean(x)
    return np.clip(x, mean-126, mean+127)

def low_clip(x):
    return np.clip(x, 255, 65535)

def minmax(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def max_normalise(x):
    max = np.max(x)
    return x / max

# change this function to another pre-processing function
# if needed
def preprocess(x):
    return max_normalise(low_clip(x))


## DATASET OPERATIONS

def even_round(num):
    return round(num/2.)*2

# need to find a way to shuffle this pairwise!
def dataset_split(dataset, labels, test_size=0.2):
    test_size = even_round(len(dataset)*test_size)
    x_train, x_test = dataset[:-test_size], dataset[-test_size:]
    y_train, y_test = labels[:-test_size], labels[-test_size:]
    return x_train, x_test, y_train, y_test


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
        if label not in [1, 12, 13, 24] and letter in ["H", "P"]:
            return True
    else:
        print("No CK found")
    return False


def filenames_to_labels(filenames, folder="/Users/Leonore/Documents/Workspace/l4proj/data/processed/"):
    # 0: unstimulated
    # 1: OVA
    # 2: ConA
    # 3: empty
    labels = []
    DMSO = []

    for file in filenames:
        labels.append(get_label(file))
        if is_dmso(file):
            DMSO.append(len(labels))
    return labels, DMSO


def read_folder_filenames(folder):
    return [os.path.join(folder, f)for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f[0] != '.' and "Brightfield" not in f]


def resize_images(folder='/Users/Leonore/Documents/Workspace/l4proj/data/raw/',
                  dst="/Users/Leonore/Documents/Workspace/l4proj/data/processed/CK22/", w=200, h=200):
    filenames = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for file in filenames:
        if not os.path.exists(os.path.join(dst, file.replace(folder, ''))):
            try:
                image = imread(file)
                image_resized = center_crop(image, w)
                img = reformat(image_resized)
                img.save(dst + file.replace(folder, ''))
            except Exception as e:
                print(e)
                print("{} is causing issues".format(file))


def images_to_dataset(folder="/Users/Leonore/Documents/Workspace/l4proj/data/processed/", w=192, h=192, process=True):
    filenames = []
    for ck in ["CK19", "CK21", "CK22"]:
        filenames.extend(read_folder_filenames(folder+ck))
    filenames = sorted(filenames)
    dataset = np.ndarray(shape=(len(filenames), w, h), dtype=np.float32)
    i = 0
    for file in filenames:
        image = imread(file)
        image_resized = center_crop(image, size=w)
        if process:
            image_resized = preprocess(image_resized)
        dataset[i] = image_resized
        i += 1
    print("All files formatted into dataset.")
    return dataset, filenames


def sliding_window(img, dest_size):
    new_img = np.full_like(img, img)

    size = img.shape[0]
    if dest_size > size or dest_size % 2 != 0:
        raise Exception("destination size is bigger than picture size or destination size is not even")

    qty = size//dest_size
    if size % dest_size != 0:
        # need to crop out the left and bottom (less significant in dataset)
        crop = size-dest_size*qty
        new_img = new_img[crop:, :-crop]

    windows = np.ndarray(shape=(qty**2, dest_size, dest_size), dtype=np.float32)

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
