import argparse
import numpy as np

import keras
from keras.models import load_model
from keras.models import Model
import umap.umap_ as umap

from cell_autoencoder import make_autoencoder, evaluate
from plot_helpers import plot_clusters, plot_live
from dataset_helpers import get_label, preprocess

"""
Main tool to run image analysis
"""

parser = argparse.ArgumentParser(description='Train model with input dataset, evaluate.')

parser.add_argument('--input', '-i', action='store', type=str, help='Compressed NPZ file to process images from', required=True)
parser.add_argument('--weights', '-w', action='store', nargs=2, type=str, help='Two h5 files for decoder and encoder trained models (decoder.h5, encoder.h5)', required=False)
parser.add_argument('--live', '-l', action='store_true', help='Make clustering visualisation live')

args = parser.parse_args()
input = args.input
weights = args.weights
live = args.live

# get input
print("{} is being loaded".format(input))
npzfile = np.load(input)
x = npzfile['x'] # images
filenames = npzfile['y'] # filenames

# get labels
# TODO generalise for 3 datasets
y = [get_label(f) for f in filenames]

# preprocess
print("Preprocessing dataset")
x_combined, y_combined = preprocess(x, y)

# make autoencoder
if weights:
    print("Decoder file is {}".format(weights[0]))
    decoder = load_model(weights[0])
    print("Encoder file is {}".format(weights[1]))
    encoder = load_model(weights[1])
else:
    # not recommended on local - could take hours
    decoder, encoder = make_autoencoder()
    train(decoder, x_train)

# evaluate
# evaluate(decoder, x_combined, x_combined)

# run clustering
print("Clustering encoded images")
encoded_imgs = encoder.predict(x_combined)
x_umap = umap.UMAP().fit_transform(encoded_imgs, y_combined)
if args.live:
    print("Visualisation being plotted live...")
    plot_live(x_umap, y_combined, x_combined)
else:
    plot_clusters(x_umap, y_combined)
