# Code.md

## Structure overview

```
.
├── clustering.py               <-- functions for running UMAP + t-SNE
├── compress_files.ipynb        <-- notebook to preprocess images into a dataset
├── config.py                   <-- /!\ config file for variables re-used across the project
├── dataset_helpers.py          <-- helper functions commonly used to process images
├── evaluation_helpers.py       <-- functions used for evaluation (plots etc.)            
├── models.py                   <-- code for initialising deep learning models, training them, and evaluating them
├── readme.md                   <-- this file!
├── run_evaluation.ipynb        <-- notebook to load or train models and evaluate them
└── segmentation.py             <-- functions used to perform image segmentation and calculate segmentation metrics
```

## Build instructions

### Requirements

* Python 3.7
* Packages: listed in `requirements.txt`

**N.B. This project runs on Keras with Tensorflow 1.15. Code is likely to break with Tensorflow 2.0 and needs to be adapted for it**

### Build steps

```bash
python3 -m venv .venv
source .venv/bin/activate && pip install -r requirements.txt
```

### Test steps

This was a research project and is not really 'built', but it can be evaluated on sample data by running the cells in the `compress_files` and `run_evaluation` notebooks. Feel free to reuse any code written here, but it was developed with a specific dataset with specific labels in mind.

```bash
jupyter notebook
# navigate to localhost:8888/tree/
```
