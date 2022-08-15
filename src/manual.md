# User manual

This .md describes how to re-use some of the code available here.

### Deep learning models

There is code for initialising an autoencoder model and a deep neural network for regression in `models.py`. The regression model depends on the autoencoder model. Input settings depend on variables in `config.py`.

### Data pre-processing

There is a useful sliding window function in the `compress_files.ipynb` notebook which transforms high dimensional images into smaller patches of images.

### Live visualisation of projection graphs

Algorithms like UMAP and t-SNE project a high dimensional dataset to a 2D or 3D projection. The `plot_live` function in `evaluation_helpers.py` transforms this visualisation to a live tool. It's a bit of a hack compared to tensorboard which I could not get to work for my dataset. It won't be as efficient for larger datasets, but is pretty fast for standard datasets like MNIST. It's displayed in a matplotlib window so the usual plot dimension tuning and zoom tools are available too.

The code was adapted from https://stackoverflow.com/a/58058600

The general plotting function used for plotting UMAP or t-SNE is also available in `evaluation_helpers.py` as `plot_clusters` and is pretty standard to use and can be applied to different datasets. Just change the labels. 

### Make a GIF from a t-SNE visualisation

The notebook in `/data/notebooks/make_gif.pynb` gives some code to make a gif from the process of building a t-SNE visualisation out of a high dimensional dataset. It's pretty.

The code was adapted from https://github.com/oreillymedia/t-SNE-tutorial

### Greyscale image segmentation

You can obtain a binary mask of a greyscale image using the `get_mask` function in `segmentation.py`. The `threshold` function in the same file attempts to achieve the same thing, but will not work as well for all datasets.
