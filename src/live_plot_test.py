import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from keras.datasets import mnist
import umap.umap_ as umap

from evaluation_helpers import plot_live

(_, _), (x_test, y_test) = mnist.load_data()

size = 9800

x_test = x_test[:size]
x_umap = umap.UMAP().fit_transform(x_test.reshape(size, 28*28))
y = np.array(y_test[:size])
labels = [x for x in range(10)]

plot_live(x_umap, y, x_test, labels)
