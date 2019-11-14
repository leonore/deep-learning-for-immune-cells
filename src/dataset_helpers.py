## Standard imports
import numpy as np
import os
from skimage.io import imread, imsave, imread
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image

# IMAGE FORMATTING OPERATIONS

def reshape(img, w=200, h=200):
    return np.reshape(img, (w, h))

def reformat(img):
    formatted = (img).astype('uint16')
    return Image.fromarray(formatted)

def center_crop(img, size=200):
    to_crop = (img.shape[0]-size)/2
    image_resized = skimage.util.crop(img, (to_crop, to_crop))
    return image_resized

def normalise(img):
    # normalise 16-bit TIF image
    return img / 65535.0

## DATASET OPERATIONS

def even_round(num):
    return round(num/2.)*2

# need to find a way to shuffle this pairwise!
def dataset_split(dataset, labels, test_size=0.2):
    test_size = even_round(len(dataset)*test_size)
    x_train, x_test = dataset[:-test_size], dataset[-test_size:]
    y_train, y_test = labels[:-test_size], labels[-test_size:]
    return x_train, x_test, y_train, y_test

def filenames_to_labels(filenames, folder="/Users/Leonore/Documents/Workspace/l4proj/data/processed/"):
    # 0: unstimulated
    # 1: OVA
    # 2: ConA
    # 3: empty
    labels = []
    DMSO = []
    for file in filenames:
        # file format: folder/CKX - L - 00(...)
        file = file.split("(")[0].replace(folder, '')
        # get letter for DMSO indices
        letter = file.split('-')[1].strip()
        label = file[-2:].strip()
        ck = file[:4]
        if ck == "CK19":
            if label in ["5", "8", "11", "15", "18", "21"] and letter in ["N", "O", "P"]:
                DMSO.append(len(labels))
            if label in ["3", "4", "5", "6", "7", "8", "24"]:
                label = 0
            elif label in ["9", "10", "11", "13", "14", "15", "23"]:
                label = 1
            elif label in ["16", "17", "18", "19", "20", "21", "22"]:
                label = 2
            else:
                label = 3
        elif ck == "CK21" or ck == "CK22":
            if label in ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]:
                label = 0
            elif int(label) in range(14, 24):
                label = 2
            else:
                label = 3
            if label != 3 and letter in ["H", "P"]:
                DMSO.append(len(labels))
        else:
            print("No CK found")
        labels.append(label)
    return labels, DMSO

def read_folder_filenames(folder):
    return [os.path.join(folder, f)for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f[0] != '.']

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

def images_to_dataset(folder="/Users/Leonore/Documents/Workspace/l4proj/data/processed/", w=200, h=200):
    filenames = []
    for ck in ["CK19", "CK21", "CK22"]:
        filenames.extend(read_folder_filenames(folder+ck))
    filenames = sorted(filenames)
    dataset = np.ndarray(shape=(len(filenames), w, h), dtype=np.float32)
    i = 0
    for file in filenames:
        try:
            image = imread(file)
            dataset[i] = normalise(image)
        except Exception as e:
            print(e)
            print("{} is causing issues".format(file))
        i += 1
    return dataset, filenames

dataset, filenames = images_to_dataset()
