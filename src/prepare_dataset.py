import os
import shutil
from PIL import Image

def duplicate_directory(src, dst):
    try:
        shutil.copytree(src, dst)
    except Exception as e:
        if e.errno == 17:
            print("Found existing data/processed directory, overwriting")
            shutil.rmtree(dst)
            duplicate_directory(src, dst)
        else:
            print('Could not copy directory structure. Error: %s' % e)

def convert(dataset_path):
    for _, _, files in os.walk(dataset_path):
        for filename in files:
            filepath = os.path.join(dataset_path, filename)
            image = Image.open(filepath)
            image = image.convert(mode="RGB")
            image.save(filepath + 'my.jpeg')

def resize(dataset_path, new_width, new_height):
    for _, _, files in os.walk(dataset_path):
        for filename in files:
            filepath = os.path.join(dataset_path, filename)
            img = Image.open(filepath)
            img = img.convert('RGB')
            img.thumbnail((int(new_width), int(new_height)))
            filepath = filepath.rstrip('.tif')
            img.save(filepath + '.jpeg', 'JPEG')

duplicate_directory('../data/raw', '../data/processed')
convert('../data/processed')
