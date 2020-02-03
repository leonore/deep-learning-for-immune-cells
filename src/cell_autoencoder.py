import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, copy

from sklearn.manifold import TSNE
import umap

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, PReLU
from keras.models import Model

from dataset_helpers import minmax, is_faulty
from dataset_helpers import get_label
from plot_helpers import show_image, plot_range
from segmentation import threshold, get_mask


imw = 192
imh = 192
c = 3
RS = 2211
