# Readme

### Jupyter notebooks
* `cell_autoencoder.ipynb` includes the autoencoder model and clustering algorithm being tuned
* `baseline_performance.ipynb` reports on the clustering algorithm's base performance with unmodified datasets
* `cell_histograms.ipynb` contains research on the structure of the dataset for better pre-processing
* `compress_files.ipynb` is the code used to compress images into efficient NPZ files for re-use locally and on Google Colab
* `visualise_tsne.ipynb` is the code used to visualise step-by-step t-sne clustering, returning a GIF
* `clustering_mnist.ipynb` contains the tutorial code for clustering MNIST with PCA, PCA+t-sne, and UMAP

### Helper files
* `prepare_dataset.py` contains the code used to transfer the needed images from the OneDrive dataset for transformation
* `dataset_helpers.py` contains helper functions to prepare the dataset
* `plot_helpers.py` contains helper functions to plot different results


## Build instructions

**You must** include the instructions necessary to build and deploy this project successfully. If appropriate, also include
instructions to run automated tests.

### Requirements

* Python 3.7
* Packages: listed in `requirements.txt`

### Build steps

```bash
pip install -r requirements.txt # if in this directory
```

### Test steps

List steps needed to show your software works. This might be running a test suite, or just starting the program; but something that could be used to verify your code is working correctly.

Examples:

* Run automated tests by running `pytest`
* Start the software by running `bin/editor.exe` and opening the file `examples/example_01.bin`
