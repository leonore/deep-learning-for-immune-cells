# COPY ONEDRIVE DATA
# TO LOCAL DATA FOR LOCAL USE

import os
from shutil import copyfile

def copy_to(dataset_folder):
    ignored = 0
    data_folder = "/Users/Leonore/Downloads/OneDrive_2_07-11-2019"
    files = [f for f in os.listdir(data_folder)]
    for filename in files:
        src = os.path.join(data_folder, filename)
        dst = os.path.join(dataset_folder, filename)
        if "Brightfield" not in filename and "tif" in filename:
            copyfile(src, dst)
        else:
            ignored += 1
            print(filename + " ignored")
    print("{} were ignored, not appropriate data".format(ignored))

copy_to('/Users/Leonore/Documents/Workspace/l4proj/data/raw/')
