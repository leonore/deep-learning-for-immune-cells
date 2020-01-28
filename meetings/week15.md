# Week 15 (2) meeting minutes - Wed 22nd January 2020

- I found that issues I was having with NaN values came from 255-clipping images that did not have values above 255; upon further inspection I realised these images were "faulty" e.g. water blobs
  - I had to incorporate this in my image processing

- I showed that masking out the background from images got really good results for getting unstimulated images as a cluster

- I found a way to threshold images as a way to get masks in a much faster way than with K-means. Using the standard deviation and the mean, I can clip values in such a way that I get similar masks as with K-means.
  - I need to compare this performance.

- For the next week I need to liaise with Hannah in order to get the data I need for evaluation, as well as finalise the deliverables for the autoencoder and preprocessing.
