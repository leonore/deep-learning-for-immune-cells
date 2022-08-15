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

## Week 7 c/ 4 November 2019 (11.5/15 hours)

### 6 November 2019
- *0.5 hour* Reading an article comparing UMAP performance to tsne

### 7 November 2019
- *1 hour* Reading about t-sne and writing down notes for dissertation
- *1 hour* Following tutorials on PCA, t-sne, UMAP to get baseline performance with digit dataset
- *3.5 hours* Moving whole datasets from OneDrive for local processing

### 8 November 2019
- *2 hours* Listening to a lecture about convolutional neural networks and autoencoders, and PyTorch
- *0.5 hour* Reflecting on best way to get all labels parsed from filenames

### 9 November 2019
- *1 hour* Rewriting and testing function to read filenames into labels for 3 CKXX datasets
- *0.5 hour* Looking at best way to parse DMSO filename index values

### 10 November 2019
- *0.5 hour* Building DMSO dataset and labels (can be improved)
- *0.5 hour* Testing own code on MNIST dataset to see if it works
- *0.5 hour* Testing clustering code on DMSO dataset

## Week 8 c/ 11 November 2019 (14/15 hours)

### 12 November 2019
- *0.5 hour* Testing clustering code on full dataset

### 14 November 2019
- *1.5 hour* Reorganising code into different files
- *1 hour* Tuning autoencoder for dimensionality reduction; trying new models  

### 15 November 2019
- *2 hours* Reviewing research done on autoencoders for dimensionality reduction and using CNNs for cell classification
- *2 hours* Moving code to Colab
- *3.5 hours* Visualising filters, scaling data, tuning autoencoder, attempting to get better results.
- *1 hour* Plotting model filters; seeing where data is being lost in the autoencoder

### 17 November 2019
- *2.5 hours* Resizing images; trying new autoencoder architecture

## Week 9 c/ 18 November (10/15 hours)

### 18 November 2019
- *1.5 hours* Trying each of: resizing data to smaller images, scaling data, adding normalisation post-activation, relu vs leakyrelu
- *1 hour* Tuning; creating visual presentation of tweak results
- *1.5 hour* Trying to see where model is going wrong with MNIST dataset

### 19 November 2019
- *3 hours* Preprocessing images with edge detectors, contrast tuning, choosing best filters

### 20 November 2019
- *2 hours* Using CIFAR autoencoder on dataset; visualising feature maps activations to pick out useful layers

### 21 November 2019
- *1 hour* Running clustering code on full + DMSO dataset to have baseline

## Week 10 c/ 25 November (10.5/15 hours)

### 27 November 2019
- *1 hour* Debugging autoencoder code with empty convolutional feature maps
- *1 hour* Exploring the encoder model; looking at weights

### 28 November 2019
- *1 hour* Building encoder from trained decoder, comparing model weights

### 30 November 2019
- *1.5 hour* Making evaluation script to run every time a change is made to a model
- *1 hour* Evaluating different models and running it with the CIFAR dataset to see where the issue lies

### 1 December 2019
- *1 hour* Researching image augmentation
- *1 hour* Looking at other small dataset + autoencoder performance for comparison
- *1 hour* Visualising features with different activation functions
- *1.5 hour* Pairing up tcells/dcells into one image before training, testing code on that
- *0.5 hour* Playing with dataset

## Week 11 c/ 2 December (18.5/15 hours)

### 2 December 2019
- *1 hour* Playing with dataset (cropping, shuffling) to see if it impacts learning
- *1 hour* Doing image augmentation
- *0.5 hour* Researching sigmoid function and testing model with different activation functions + MNIST

### 3 December
- *2 hour* Looking at pre-processing steps for debugging
- *1.5 hour* Seeing if output is better with new pre-processing + old models
- *0.5 hour* Debugging models again

### 5 December
- *1.5 hour* Working on clipping images to examine impact on autoencoder training

### 6 December
- *0.5 hour* Talking to Carol about best ways to pre-process images
- *0.5 hour* Reporting on and cleaning up work on pre-processing images differently
- *0.5 hour* Clustering new encoded images with t-sne
- *1 hour* Working on a sliding window function

### 7 December
- *4 hours* Transferring OneDrive cell images to hard drive for easier access

### 8 December
- *1.5 hour* Getting full DMSO dataset with sliding window function
- *2.5 hour* Using NPZ files to save compressed versions of the dataset

## Week 12 c/ 9 December (15.5/40 hours)

### 9 December
- *0.5 hour* Analysing cell histograms more in-depth
- *0.5 hour* Re-transferring DMSO dataset, but sorted
- *0.5 hour* Double checking files correspond to labels, sorting is right

### 10 December
- *1 hour* Meeting about cell images, double-checking the data afterwards
- *2 hours* Lecture on evaluation and dissertation writing

### 11 December
- *1 hour* Loading unmodified DMSO files into Google Drive
- *1 hour* Looking at clipping pixel values below 255

### 12 December
- *1 hour* Evaluating performance with values below 255 clipped and cleaning up
- *1 hour* Clustering 255-clipped dmso dataset

### 13 December
- *1 hour* Modifying t-sne visualisation code to add legend
- *1 hour* Bugfixing, adding baseline performance of DMSO processed
- *2.5 hours* Working on visualising step by step t-sne clustering
- *0.5 hour* File cleanup
- *1.5 hours* Comparing performance on autoencoder between 2 preprocessings, tuning autoencoder with leakyrelu
- *0.5 hour* Running t-sne visualisation on DMSO dataset

## Week 13 c/ 16 December (25/40 hours)

### 16 December
- *1.5 hours* Tuning model and clustering results
- *1 hour* Reading about image segmentation with K-means
- *0.5 hours* Experimenting to calculate overlaps/means/etc from mask of images
- *1 hour* Reading and trying out intersection over union  
- *2 hours* Image segmentation with K-means
- *0.5 hours* Writing function to combining to overlap windows from dataset

### 17 December
- *2.5 hours* Trying to compress combined DMSO images: not very successful, computer lacks RAM
- *1 hour* Writing pseudo-code for overlapping images: combining will be done via arrays rather than files
- *1.5 hours* Analysing K-means overlap of images
- *1.5 hours* Working on combining DMSO images into 2-channel image in Google Colab, re-using previous overlap code

### 18 December
- *2 hours* Tuning T-SNE with different parameters to see if clustering of 2-channel DMSO changes
- *0.5 hours* Evaluating K-means overlaps with matplotlib
- *3 hours* Collating all explored processing methods, image combinations, into one cell_autoencoder notebook to condense autoencoder research

### 19 December
- *1 hour* Finishing collating all explored processing methods into one cell_autoencoder notebook
- *0.5 hours* Removing redundant code and files for cleanliness
- *1 hour* Looking at structural similarity index for exploring effects of drugs
- *1.5 hour* Running autoencoder on cell images with not-cell pixels masked out (obtained from K-Means)
- *2.5 hours* Writing status report

-------------

## Christmas break (4.5 hours)


### 26 December   
- *1 hour* Reading about image preprocessing and which technique is best suited ([0,1], [-1,1], etc.)

### 27 December
- *1 hour* Reading about UNet and following a tutorial to build a simple model

### 30 December
- *1 hour* Trying to run UNet model with own dataset, struggling with speed of getting mask

### 2 January
- *0.5 hours* Training UNet model with own dataset, getting good results
- *1 hour* Reading about image preprocessing and which technique is best suited

------------

## Semester 2
## Week 1 (14) c/ 13 January (5.5/15 hours)

### 14 January
- *2.5 hours* Testing image processing methods

### 17 January
- *0.5 hours* Writing down some resources for the dissertation and writing down meeting minutes for week 14

### 18 January
- *1 hour* Working on thresholding as an image segmentation technique, potentially much faster than K-Means
- *1 hour* Testing autoencoder with batch normalisation, average pooling, removing bias
- *0.5 hours* Reading about why upsampling (resize) + Conv2D is preferred to Conv2DTranspose

## Week 2 (15) c/ 20 January (22.5/15 hours)

### 20 January
- *0.5 hours* Looking at outlier images to best code image pre-processing
- *4.5 hours* Researching autoencoder parameters and testing models
- *0.5 hours* Debugging NaN error in processing with values < 255 clipped
- *1 hour* Visualising autoencoder results and feature maps to see when it stops learning new features, debugging visualisation function
- *0.5 hours* Discarding outlier images to see if clustering improves for distinguishing OVA and ConA

### 21 January
- *1 hour* Working on removing faulty images from the dataset

### 22 January
- *2.5 hours* Working on removing faulty images from the dataset, lots of debugging
- *1.5 hours* Debugging processing code, low_clip returns absurd values... or is it minmax?
- *0.5 hours* Writing main preprocessing function

### 23 January
- *1 hour* Failing to preprocess images with K-means masking out the background, takes too long
- *1.5 hours* Working on plotting the indices of image outliers in clusters to see if the outliers make sense

### 24 January
- *1 hour* Working on pushing code from Google Colab and polishing autoencoder file to be a deliverable
- *2 hours* Using new visualisation function on autoencoder model and datasets
- *1 hour* Working on optimising K-Means function as working on the full dataset runs out of memory in Colab

### 26 January
- *0.5 hours* Researching different options for hover visualisation of index values of plotting points for easier outlier detection
- *1.5 hours* Giving references for autoencoder decision, doing research on past work with biomedical data
- *1 hour* Having fun with autoencoder structures
- *0.5 hours* Polishing processing deliverable

## Week 3 (16) c/ 27 January (14/15 hours)

### 27 January
- *2 hours* Collating research metrics for all datasets in data/target from data/raw
- *0.5 hours* Writing evaluation plan
- *1 hour* Comparing OpenCV vs. sklearn k-means performance, thresholding vs. k-means performance, finishing up segmentation code
- *0.5 hours* Looking for different possible ways of hovering over plots to see indexing values
- *0.5 hours* Looking at UNet model again with faster K-means

### 28 January
- *0.5 hours* Writing up meeting minutes for two weeks (forgot to do last week's somehow)

### 29 January
- *1.5 hours* Compressing files (using np.int16) and cleaning up Drive to make space for large files

### 30 January
- *1 hour* Cleaning up Google Drive to upload datasets, uploading datasets (Total ~15Go on Drive)
- *1 hour* Reading up on issues with image processing in biological researcher; looking into flat field correction
- *0.5 hours* Looking at dataset, experimenting with image correction
- *1 hour* Writing some dissertation bullet points, getting ideas in order. Issues with setting out the problem
- *0.5 hours* Looking at the biological research done as provided by Hannah

### 31 January
- *1.5 hours* Visiting immunology building to look at current software being used for cell image analysis and writing up notes
- *1.5 hours* Bulletpoint dissertation writing, additional background research on previous uses of image segmentation
- *0.5 hours* Writing out (on paper) a restructured dissertation plan; using materials and methods instead of analysis and design

## Week 4 (17) c/ 3 February (19/15 hours)

### 3 February
- *4 hours* Working on live visualisation outliers with hover annotations in matplotlib, struggling with getting all scatter points included
- *1 hour* Working on a reconstruction function from the separated images
- *2 hours* Jupyter notebooks to Python files (if applicable), moving notebooks to data/

### 4 February
- *1.5 hours* Training model with DMSO dataset again to see if weights can be reused on other datasets
- *1 hour* Research about benefits of deep networks with classification for high dimensional data
- *2.5 hours* Looking into background subtraction instead of flat field correction
- *1 hour* Working on visualising outlier images alongside the plot for the outlier plotting

### 5 February
- *1.5 hours* Finishing up cell_autoencoder functions, creating main file
- *1 hour* Adding an image visualisation to the live cluster plotting function, works with MNIST

### 9 February
- *0.5 hour* Working on main interface to launch evaluations in main.py, running into local memory troubles so adapting preprocessing function a bit
- *2.5 hours* Debugging main.py + associated files and writing bullet points in materials & methods for dissertation
- *0.5 hours* Final tweaks to live visualisation plot

## Week 5 (18) c/ 10 February (23.5/15 hours)

### 10 February
- *0.5 hours* Cleaning up notebooks and adding baseline performance for combined dataset
- *1 hour* Finishing up presentation, preparing Colab notebook for obtaining weight, testing demo for meeting tomorrow
- *1 hour* Working on get_labels function for all 3 datasets, running into troubles with filenames formatting (01 instead of 1)

### 11 February
- *1 hour* Researching uses of t-sne, looking into building a classifier as clusters are formed
- *1 hour* Getting masked autoencoder weights from Colab and building a classifier from the encoder model
- *1 hour* Training model, getting results - plotting confusion matrix
- *2 hours* Training on full CK19 to try mitigate unbalance in classes, have to fix preprocessing function (%200 check doesn't work)
- *0.5 hours* Fixing an additional bug with minmax when min=max
- *0.5 hours* Training autoencoder on balanced dataset, looking at tensorboard visualisation
- *0.5 hours* Writing code to compare overlaps computed by GE software vs. K-means segmentation with IoU
- *0.5 hours* Working out kinks between CK19 filenames and CK19 metrics, noticed missing files

### 13 February
- *0.5 hours* Having to modify NPZ file because of errors in copied missing file, differences are now resolved
- *1 hour* Debugging overlap calculations code, now accounting for black images
- *2.5 hours* Running calculations, making adjustments – scores seems way off
- *1 hour* Trying new overlap metrics, not IoU --> e.g. Dice. Asking Hannah about interaction measure- area % or count where there is overlap?

### 14 February
- *1.5 hours* Working on image reconstruction; checking everything is OK.
- *1.5 hours* Deciding on metric = IoU/total area of overlap to account for subimages which might be 0; running calculations
- *0.5 hours* Comparing results with target results; histograms show promise
- *0.5 hours* Adapting plot; thinking about removing faulty images for classification for accuracy
- *1 hour* Running final code and saving metrics and labels
- *2 hours* Adapting classifier to predict an interaction value
- *1 hour* Using a non-negative kernel constraint to prevent negative prediction; getting better results

### 16 February
- *1 hour* Tuning classifier

## Week 6 (19) c/ 17 February (11/15 hours)

### 19 February
- *0.5 hours* Meeting

### 22 February
- *1.5 hours* Fixing labels and improving shuffle function to handle larger datasets, see efficient_shuffle()
- *0.5 hours* Training model with shuffling, getting good results
- *1 hour* Working on visualisation with varying size scatter points + colour maps

### 23 February
- *0.5 hours* Going over meeting and writing down meeting minutes
- *1 hour* Running regression code, seems to be really good results but thinking of best way to plot performance.
- *1 hour* Looking at tools to evaluate regression performance. Lots of 0 values might create a skew. Looking at past research
- *0.5 hours* Looking at how UNet performs in term of overlap calculations; however it adds multiple extra steps
- *1 hour* Reviewing past research on combining bottleneck autoencoder layer with regression models
- *0.5 hours* Trying to remove values <0 for predictions for plotting
- *1 hour* Realising big mistake in code: UMAP has been running the clustering supervised. running visualisation tests again to check if clusters are still somewhat discernible
- *2 hours* Getting new weights with well-shuffled CK19 dataset; t-sne/UMAP indeed doesn't find clear clusters

## Week 7 (20) c/ 24 February (22/15 hours)

### 24 February
- *5 hours* Fixing big loss stagnation issues when trying to obtain weights from regression model; was because of MAE metric and probably overwhelming 0s in dataset
- *1 hour* Running training now that stagnating loss is fixed; starting a bash script for evaluation
- *2 hours* Making functions to evaluate regression performance
- *1 hour* Writing bits for 'materials & methods' in dissertation

### 25 February
- *1 hour* Summarising for tomorrow's presentation
- *2 hours* Waiting for trained models with final attempts at tuning: regression dropout, smaller autoencoder. callbacks with both.
- *1 hours* Obtained final weights with smaller model and tuned layers, but had to stop evaluating because Colab restricted accelerated usage
- *0.5 hours* Finishing up evaluation script
- *1.5 hours* Refactoring main.py file and associated components
- *1 hour* Making regression model file, waiting for autoencoder to train for masked balanced dataset

### 26 February
- *0.5 hours* Refactoring calculate_overlaps file
- *1 hour* Meeting and writing meeting artifacts
- *2 hours* Refactoring code. Not sure whether to make it all Python or keep Jupyter notebooks
- *1 hour* Seeing if I can put all processed images in one file, but it's too big for Colab
- *1 hour* More refactoring, getting final weights and putting them in data/

### 28 February
- *0.5 hours* Going over collected research to develop a structure for "background" dissertation

## Week 8 (21) c/ 2 March (28/15 hours)

### 2 March
- *1 hour* Moving evaluation to a Colab notebook, running it locally was taking its toll on my laptop.
- *1 hour* Running evaluation and debugging while writing dissertation
- *2 hours* Working on dissertation structure to discuss at meeting tomorrow, drafting materials & methods

### 3 March
- *0.5 hours* Meeting and writing down artifacts
- *2 hours* Writing down first draft for material & methods to send to Carol

### 4 March
- *5 hours* Running evaluation; debugging matplotlib; having trouble with models predicting different results in different sessions?

### 5 March
- *6 hours* Looking through Keras github issues to try and fix model.predict() getting massively different results across different sessions

### 7 March
- *4 hours* Realised the opencv's random k-means center initialisation was producing wildly different results for overlap calculations; recalculated; checked shuffling. Getting much better results (very good fit on the line).
- *2 hours* Writing dissertation background (also while waiting for some things to run in Colab)

### 8 March
- *3 hour* Writing dissertation background
- *4 hours* Working on reprocessing of overlaps, cleaning up files
- *1 hour* Re-training regression, trying to re-train decoder but have ran out of Colab GPU allocation

## Week 9 (22) c/ 9 March (2/15 hours)

### 9 March
- *1.5 hours* Writing dissertation; drawing system diagram
- *0.5 hours* Can finally run notebooks on Google Colab, was disconnected from GPU for usage limit
