# Week 11 meeting minutes - 5 December 2019

* I ran through the file where I had kept my progress for the past week, showing that I found why the model was not learning (wrong output activation function), but that the lack of data was not the issue, and rather the data itself might be (as models work fine for CIFAR10, MNIST).
* I showed Carol a histogram of the pixel values of a sample cell image. It highlighted that all of them are quite low valued, and more suited for an 8-bit image than a 16-bit image.
* Carol suggested I transform the values above 255 to 0 (black) and see what happens.
* If after this image transformation, which seems to be the source of autoencoder problems, the autoencoder does not seem to be improving, I can try and move on to try and do image segmentation, i.e. analyse the structure of the cell images directly and see if I can calculate some overlap with that. Here are some things to explore:
  * k-means clustering (2 clusters)
  * contour finding
  * structure similarity
  * blob detection
