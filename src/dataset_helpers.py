# Standard imports
import numpy as np
import os

from segmentation import get_mask
from config import RS, imw, imh, c, size

# IMAGE OPERATIONS

def low_clip(x):
    return np.clip(x, 255, 4095)


def minmax(x):
    if x.min() == x.max():
        return x
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def is_faulty(x):
    if x.max() <= 255:
        return True
    return False

# DATASET OPERATIONS

def efficient_shuffle(*arrays, random_state=None):
    """
    shuffling function that does not run out of memory
    but shuffling is done in place
    """
    if not random_state:
        random_state = np.random.randint(0, RS)
    for arr in arrays:
        np.random.seed(random_state)
        np.random.shuffle(arr)


def even_round(num):
    return round(num / 2.) * 2


def dataset_split(*arrays, test_size=0.2):
    test_size = even_round(len(dataset) * test_size)
    results = []
    for arr in arrays:
        results.append(arr[:-test_size], arr[-test_size:])
    return results


def is_dmso(file):
    """
    hardcoded labelling helper function
    DMSO labels are specified in `data/raw/plate_layouts`
    """
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
    """
    hardcoded get_label function
    labels can be found in `data/raw/plate_layouts`
    """
    # 0: unstimulated
    # 1: OVA
    # 2: ConA
    # 3: Faulty
    # 4: empty

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
    elif ck == "CK21":
        if number in ["01", "12", "13", "24"]:
            label = 4
        elif number in ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]:
            label = 0
        elif int(number) in range(14, 24):
            label = 2
        else:
            label = 3
    elif ck == "CK22":
        if number in ["01", "12", "13", "24"]:
            label = 4
        elif number in ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]:
            label = 0
        elif int(number) in range(14, 24):
            label = 1
        else:
            label = 3
    elif ck == "CK01":
        if letter == "B":
            label = 1
        elif letter == "E":
            label = 2
        else:
            label = 3
    else:
        print("No CK found")
        return False

    return label


def remove_faulty(filenames):
    """
    function to get labels for an image
    without the `faulty` label
    this is desirable for our regression model,
    where a separate faulty category does not matter
    and messes up visualisations
    """

    l = len(filenames)
    y = np.ndarray(shape=(l // 2, ), dtype=np.uint8)

    idx = 0
    i = 0
    count = 0

    nb = (size//imw)**2
    # loop through images and process
    while idx < l - nb:
        # ignore 100, 300, etc. values as they will already have been processed
        if count == nb:
            count = 0
            idx += nb
        else:
            y[i] = get_label(filenames[idx])

            idx += 1
            i += 1
            count += 1

    print('Faulty labels removed. Size of labels: {}'.format(l // 2))
    return y


def read_folder_filenames(folder):
    """
    return filepaths for the images we want
    """
    return [os.path.join(folder, f)for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f[0] != '.' and "Brightfield" not in f]


def reconstruct_from(images, imw=imw, size=size):
    """
    this reverses the sliding_window operation
    if we want to visualise an original image
    @imw: the width of the subimage
    @size: the size of the original image
    --> nb: the number of subimages created per image
    """
    nb = size // imw
    new_img = np.ndarray(shape=(imw * nb, imw * nb), dtype=np.float32)
    y = 0
    x = 0
    for i in range(nb**2):
        new_img[x:x + imw, y:y + imw] = images[i]
        x += imw
        if x == imw * nb:
            x = 0
            y += imw
    return new_img


def combine_images(data, filenames, mask=False):
    """
    combine_images combines separate B&W images in an RGB image
    each image's counterpart is found X images later in the array
    where X = (size of the original image / size of the sub image)**2 --> number of sub-images per image
    """

    l = len(data)

    # initialise arrays for filling in
    x_data = np.ndarray(shape=(l // 2, imw, imh, c), dtype=np.float32)
    y_data = np.ndarray(shape=(l // 2, ), dtype=np.uint8)

    # initialise index values
    idx = 0
    i = 0
    count = 0

    full = (size//imw)**2  # number of sub-images per full image (well)

    # loop through images and process
    while idx < l - full:
        # ignore 100, 300, etc. values as they will already have been processed
        if count == full:
            count = 0
            idx += full
        else:
            # if the image is "faulty" we cannot low_clip and apply minmax -> NaN
            if is_faulty(data[idx]) or is_faulty(data[idx + full]):
                x_data[i, ..., 1] = minmax(data[idx])
                x_data[i, ..., 0] = minmax(data[idx + full])
                y_data[i] = 3
            else:
                x_data[i, ..., 1] = minmax(low_clip(data[idx]))
                x_data[i, ..., 0] = minmax(low_clip(data[idx + full]))
                y_data[i] = get_label(filenames[idx])

            # mask out the background
            if mask:
                # red-coloured
                x_data[i, ..., 0] *= get_mask(x_data[i, ..., 0])
                # green-coloured
                x_data[i, ..., 1] *= get_mask(x_data[i, ..., 1])

            # try and save memory
            data[idx] = 0
            data[idx + full] = 0

            idx += 1
            i += 1
            count += 1

    print('Images preprocessed. Size of dataset: {}'.format(len(x_data)))
    return x_data, y_data
