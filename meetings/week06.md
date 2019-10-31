# Week 6 meeting minutes - 31 October 2019

- I showed the progress I made on:
  - building an autoencoder that's able to learn a good reconstructed image from the compressed version of the image.
  - calulating overlap of images
  - calculating label of images
- However, the clustering attempted on the images has not been getting any good results. Carol suggested trying a tutorial with PCA+tsne on the MNIST dataset, in order to make sure that I can at least get meaningful results with that.
- Hannah suggested I try the clustering on the cells that have vehicle control, as it is for those images the difference in overlap should be more visible.
- I will work on getting a clustering algorithm to work, and if it does, I can try tune the autoencoder to use lower dimensions for the compressed representation etc. in order to gain performace.
- For the remainder of the week I am first going to work on making sure I can write the resized images back to disk so I can make use of Colab's GPU and/or the university computer cluster more efficiently.
