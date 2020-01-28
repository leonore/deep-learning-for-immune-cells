# Week 14 meeting minutes - Wed 15th January 2020

- First I explained that it seems that most of my troubles with the autoencoder not working came from performing the same division operations on the dataset multiple times inadvertently because of pass-in arrays and unclear documentation in Numpy. I have been taking care now for this not to happen again.

- I explained that I had spent some time over the break researching the best way to preprocess images. Before the break, I uploaded a collation of techniques and how they performed with the autoencoder on Github.
  - However, I had doubts about which normalisation technique was best to use: for 16-bit, for 12-bit, max pixel value, per image/per dataset normalisation, etc.
  - After doing some reading and performing some tests, I found that using per-image max pixel normalisation worked best, as I had found that the histogram of the images varied greatly from one to the next. Hence using per-dataset normalisation did not work as well.
  - Moreover, I considered putting images in the range [-1, 1], however as we are dealing with images and sigmoid was performing well regardless I thought [0,1] would save some hassle.
  - The images are also 12-bit images formatted as 16-bit.  

- For my final image preprocessing techniques, I have decided to use per-image max pixel normalisation, with the two cells combined in an RGB image, and then another technique where the masks obtained from the image segmentation technique would be used to mask out the background and hopefully get even better results.
  - I tried this over the break with a sample of the dataset and it performed the best in terms of getting separate clusters.

- Carol was happy with how K-means performed for getting masks out of the cell images. Because it works so well, we can use this as a final technique, but I want to try thresholding as well just in case it performs just as well, as it will probably be faster than K-means to rum.
  - Once these masks are obtained we can calculate overlap with intersection over union and compare the results obtained with the overlap numbers obtained through GE's proprietary software used by the researcher.

- Finally, I found that using the UNet model with the masks obtained with K-means worked well for predicting masks. This could be helpful to researchers as the propriety software they use struggles to find the edges of dendritic cells.

- Moreover, the researchers do not know how to get the mask images out of the propriety software, so I could provide it to them with this.

- For the rest of the semester, I should hopefully be able to stay on track with the planned schedule from my status report.
