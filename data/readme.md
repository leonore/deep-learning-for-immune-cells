## What data can I find here?

```
.
├── evaluation                            <-- contains all outputs of evaluations ran on deep learning models
│   ├── autoencoder                       <-- autoencoder reconstruction of immune cell images
│   ├── clustering                        <-- UMAP projections
│   └── regression                        <-- regression results
|
├── notebooks                             <-- contains all notebook code that was not crucial for src
│   ├── classifier.ipynb                  <-- attempt at building a classifier for our data
│   ├── image_segmentation.ipynb          <-- exploration of image segmentation techniques
│   ├── make_gif.ipynb                    <-- make a gif from t-SNE projection
│   ├── performance_evaluations.ipynb     <-- performance evaluation of segmentation for diss
│   └── unet.ipynb                        <-- u-net model built from a tutorial
|
├── raw                                   <-- all the raw data used to build our dataset
│   ├── images                            <-- should contain our raw images but due to confidentiality I cannot upload them
│   └── plate_layouts                     <-- plate layouts contain the labelling information for the images
│       ├── CK19 plate layout.xlsx
│       ├── CK21 plate layout.xlsx
│       └── CK22 plate layout.xlsx
|
├── sample_data                           <-- all forms of the data I was allowed to upload
│   ├── evaluation                        <-- for evaluation outputs
│   ├── processed                         <-- for processed data
│   └── raw                               <-- for raw data
|
└── weights                               <-- Keras/Tensorflow 1.15 .h5 files for different developed models
    ├── decoder.h5
    ├── decoder_masked.h5
    ├── encoder.h5
    ├── encoder_masked.h5
    ├── regression.h5
    ├── regression_masked.h5
    └── unet.h5
```

* images are processed with the help of the `src/compress_files.ipynb` notebook
* evaluation data is generated on `src/models.py` from functions written in `src/evaluation_helpers.py`
