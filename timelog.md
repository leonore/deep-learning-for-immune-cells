# Timelog

* Deep learning for analysing immune cell interactions
* Leonore Papaloizos
* 2264897v
* Carol Webster

## Week 1 c/ 23 Sept 2019 (19/40 hours)

### 23 Sept 2019

* *0.5 hour* Kick-off meeting with supervisor
* *0.5 hour* Downloaded ImageJ Fiji for visualising cell images

### 24 Sept 2019
* *4 hours* Reading up on project guidance notes
* *1 hour* Project repository setup
* *1 hour* Wrote down week 1 minutes and summer report

### 25 Sept 2019
* *1 hour* Reading up on autoencoders

### 26 Sept 2019
* *2 hours* Trying to download the cell images
* *0.5 hour* Downloading Zotero for reference management and getting to grips with it

### 27 Sept 2019
* *2 hours* Reading about convolutional neural networks
* *1 hour* Reading about UNet
* *1 hour* Looking at cell pictures

### 28 Sept 2019
* *2.5 hours* Following a tutorial on autoencoders with keras
* *1 hour* Reading examples of UNet code with keras
* *1 hour* Preparing questions and meeting

## Week 2 c/ 30 Sept 2019 (9/15 hours)

### 30 Sept 2019
* *0.5 hours* Uploading post-meeting details
* *2 hour* Looking at how to use dataset with autoencoders
* *1 hour* Spent reading tutorials on adding clustering layer in autoencoder models

### 1 Oct 2019
* *4 hour* Working on processing raw TIF images to JPEG

### 6 Oct 2019
* *1 hour* Cleaning up code to read images into dataset
* *0.5 hour* Building (very low-res) basic encoder

## Week 3 c/ 7 October 2019 (8/15 hours)

### 8 Oct 2019
* *0.5 hour* Setting up Trello and writing post meeting details
* *0.5 hour* Trying to write out plan for the rest of the semester

### 9 Oct 2019
* *1 hour* Compiling dissertation, importing to Latex, familiarising with the format
* *0.5 hour* Reading all fluorescent images into raw dataset folder
* *0.5 hour* Writing code to display fluorescent images side to side for better visualisation of work

### 11 Oct 2019
* *1 hour* Working on autoencoding the fluorescent images

### 12 Oct 2019
* *2 hours* Working on getting the three different representations of the fluorescent images
* *1 hour* Summarising progress and writing a plan for the rest of the semester
* *1 hour* Looking at changing image size to see if it improves autoencoder performance

## Week 4 c/ 14 October 2019 (11/15 hours)

### 14 Oct 2019
* *3 hours* Tuning model and CNN layers parameters to get better output

### 15 Oct 2019
* *1 hour* Reading about why my autoencoder might be failing
* *0.5 hour* Setting up access to university's computer cluster
* *1 hour* Following a tutorial to visualise my model's filters
* *2 hours* Trying to improve model performance by tuning parameters, reading about models

### 16 Oct 2019
* *1 hour* Tuning convolutional layers parameters to try and get better results

### 17 Oct 2019
* *1 hour* Trying to move to Google Colab for GPU speedup; issues with using local files
* *0.5 hour* Tuning autoencoder layers; seeing improvement in performance
* *0.5 hour* Working on a function to display before/after images in subplots

### 20 Oct 2019
* *1 hour* Tuning autoencoder; looking at different representations to assess progress

## Week 5 c/ 21 October 2019 (13/15 hours)

### 22 October 2019
- *1 hour* Writing function to plot input/output of autoencoder to visualise progress
- *1.5 hours* Tuning autoencoder, visualising results and keeping records of parameters
- *0.5 hour* Deciding on a model, cropping images to a 200x200 center image and applying the model to it
- *0.5 hour* Writing a function to split a dataset into train, test without shuffling it to keep filenames for pairwise overlap calculations
- *0.5 hour* Trying out different ways of calculating overlap between pairs of images

### 23 October 2019
- *1 hour* Working on calculating overlap; struggling to find right method
- *0.5 hour* Watching video on t-cell/dendritic cells interaction for dissertation motivation

### 25 October 2019
- *0.5 hour* Trying out calculations with TSNE and overlapped images
- *0.5 hour* Fixing dataset split function
- *1 hour* Writing function to get coloured labels

### 26 October 2019
- *0.5 hour* Cleaning up jupyter notebook; adding headings for better structure
- *1.5 hour* Working on tsne clustering

### 27 October 2019
- *1 hour* Getting coloured labels on a plot and seeing if tnse results make sense (they don't)
- *1 hour* Trying to save resized images locally without loss of information for GPU access/Colab access
- *0.5 hour* Plotting overlap results with labels to see if labels make sense
- *1 hour* Experimenting with PCA decomposition

## Week 6 c/ 28 October 2019 (8.5/15 hours)

### 28 October 2019
- *5.5 hours* Computer cluster training

### 29 October 2019
- *0.5 hour* Recapping current work done for meeting

### 31 October 2019
- *1.5 hours* Working on writing resized images to disk

### 2 November 2019
- *1 hour* Refactoring and testing read/write functions

## Week 7 c/ 4 November 2019 (2.5/15 hours)

### 6 November 2019
- *0.5 hour* Reading an article comparing UMAP performance to tsne

### 7 November 2019
- *1 hour* Reading about t-sne
- *1 hour* Following tutorials on PCA, t-sne, UMAP to get baseline performance with digit dataset
