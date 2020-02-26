import argparse

"""
Main tool to run image analysis
- Functionality should be limited in this tool.
- It should mainly run methods provided by other components.
"""

parser = argparse.ArgumentParser(description='Evaluate a model by training with input dataset or based on a previously trained model weights.')

parser.add_argument('--input', '-i', action='store', type=str, help='Compressed NPZ file to process images from', required=True)
parser.add_argument('--metrics', '-m', action='store', type=str, help='Compressed NPZ file containing metrics captured from the images', required=True)
parser.add_argument('--weights', '-w', action='store', nargs=3, type=str, help='Three h5 files for decoder, encoder, regression trained models (decoder.h5, encoder.h5)', required=False)
parser.add_argument('--mask', action='store_false', help="Mask out the background of images using K-means")
parser.add_argument('--live', '-l', action='store_true', help='Make clustering visualisation live')
parser.add_argument('--tag', action='store', help="Tag for saving figures. If not specified the files won't be saved.")

args = parser.parse_args()
input = args.input
metrics = args.metrics
weights = args.weights
live = args.live
tag = args.tag
mask = args.mask

import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Model, load_model

import clustering.run_both
import cell_autoencoder.evaluate
import regression_model.evaluate
from cell_autoencoder import make_autoencoder
from regression_model import make_regression

from evaluation_helpers import plot_clusters, plot_live
from dataset_helpers import get_label, combine_images, efficient_shuffle

from config import RS

# get input
print("{} is being loaded".format(input))
npzfile = np.load(input)
x = npzfile['x'] # images

# preprocess
print("Combining images.")
x_combined = preprocess(x, mask=mask)

print("Metrics are being loaded from {}".format(metrics))
metrics = np.load(metrics)
y_combined = metrics['y_combined']
y_overlaps = metrics['y_overlaps']



# make autoencoder
if weights:
    print("Decoder file is {}".format(weights[0]))
    decoder = load_model(weights[0])
    print("Encoder file is {}".format(weights[1]))
    encoder = load_model(weights[1])
    print("Regression file is {}".format(weights[2]))
else:
    # not recommended on local - could take hours
    print("Training autoencoder... be aware this might take a long time")
    decoder, encoder = make_autoencoder()
    train(decoder, x_combined)
    print("Training regression model... be aware this might take a long time")
    regression = make_regression(encoder)

# evaluate
cell_autoencoder.evaluate(decoder, x_combined, x_combined, tag)

# run clustering
encoded_imgs = encoder.predict(x_combined)

print("Clustering encoded images")
x_tsne, x_umap = clustering.run_both(encoded_imgs, random_state=RS)

if args.live:
    print("Visualisation being plotted live...")
    plot_live(x_tsne, y_combined, x_combined)
    plot_live(x_umap, y_combined, x_combined)
else:
    plot_clusters(x_tsne, y_combined, tag+"_tsne")
    plot_clusters(x_umap, y_combined, tag+"_umap")

# run regression
print("Running regression using encoder model...")

y_pred = regression.predict(x_combined)
regression_model.evaluate(y_pred, y_true, tag)
