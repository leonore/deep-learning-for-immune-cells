import argparse

"""
Main tool to run image analysis
- Functionality should be limited in this tool.
- It should mainly run methods provided by other components.
"""

parser = argparse.ArgumentParser(description='Evaluate a model by training with input dataset or based on a previously trained model weights.')

parser.add_argument('--input', '-i', action='store', type=str, help='Compressed NPZ file to process images from', required=True)
parser.add_argument('--weights', '-w', action='store', nargs=2, type=str, help='Two h5 files for decoder and encoder trained models (decoder.h5, encoder.h5)', required=False)
parser.add_argument('--mask', '-m', action='store_false', help="Mask out the background of images using K-means")
parser.add_argument('--live', '-l', action='store_true', help='Make clustering visualisation live')
parser.add_argument('--label', action='store', help="Name of the files for saving figures. If not specified the files won't be saved.")

args = parser.parse_args()
input = args.input
weights = args.weights
live = args.live
label = args.label
mask = args.mask

import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Model, load_model

import clustering.run_both
from cell_autoencoder import make_autoencoder, evaluate
from plot_helpers import plot_clusters, plot_live
from dataset_helpers import get_label, preprocess, efficient_shuffle

RS=2211

# get input
print("{} is being loaded".format(input))
npzfile = np.load(input)
x = npzfile['x'] # images
filenames = npzfile['y'] # filenames
y = [get_label(i) for i in filenames] # get labels

# preprocess
print("Preprocessing dataset")
x_combined, y_combined = preprocess(x, y, mask=mask)

# make autoencoder
if weights:
    print("Decoder file is {}".format(weights[0]))
    decoder = load_model(weights[0])
    print("Encoder file is {}".format(weights[1]))
    encoder = load_model(weights[1])
else:
    # not recommended on local - could take hours
    print("Training autoencoder... be aware this might take a long time")
    decoder, encoder = make_autoencoder()
    train(decoder, x_train)

# evaluate
evaluate(decoder, x_combined, x_combined, label=label)

# run clustering
encoded_imgs = encoder.predict(x_combined)

print("Clustering encoded images")
x_tsne, x_umap = clustering.run_both(x_combined, random_state=RS)

if args.live:
    print("Visualisation being plotted live...")
    plot_live(x_tsne, y_combined, x_combined)
    plot_live(x_umap, y_combined, x_combined)
else:
    plot_clusters(x_tsne, y_combined, label=label)
    plot_clusters(x_umap, y_combined, label=label)
