import os
from shutil import copyfile, rmtree

def copy_to(dataset_folder):
    ignored = 0
    data_folder = "/Users/Leonore/University of Glasgow/Hannah Scales - Image analysis data/CK19/TCell_DCellInteraction10X_FinV_2014.07.04.12.14.45"
    files = [f for f in os.listdir(data_folder)]
    for filename in files:
        src = os.path.join(data_folder, filename)
        dst = os.path.join(dataset_folder, filename)
        if "Brightfield" not in filename and "tif" in filename:
            copyfile(src, dst)
        else:
            ignored += 1
    print("{} were ignored, not appropriate data".format(ignored))

def clean(dataset_folder):
    rmtree(dataset_folder)

clean('../data/raw/')
copy_to('../data/raw/')

# import shutil
# from PIL import Image
#
# def duplicate_directory(src, dst):
#     try:
#         shutil.copytree(src, dst)
#     except Exception as e:
#         if e.errno == 17:
#             print("Found existing data/processed directory, overwriting")
#             shutil.rmtree(dst)
#             duplicate_directory(src, dst)
#         else:
#             print('Could not copy directory structure. Error: %s' % e)
#
# def convert(dataset_path):
#     for _, _, files in os.walk(dataset_path):
#         for filename in files:
#             filepath = os.path.join(dataset_path, filename)
#             image = Image.open(filepath)
#             image = image.convert(mode="RGB")
#             image.save(filepath + 'my.jpeg')
#
# def resize(dataset_path, new_width, new_height):
#     for _, _, files in os.walk(dataset_path):
#         for filename in files:
#             filepath = os.path.join(dataset_path, filename)
#             img = Image.open(filepath)
#             img = img.convert('RGB')
#             img.thumbnail((int(new_width), int(new_height)))
#             filepath = filepath.rstrip('.tif')
#             img.save(filepath + '.jpeg', 'JPEG')
#
# duplicate_directory('../data/raw', '../data/processed')
# convert('../data/processed')
