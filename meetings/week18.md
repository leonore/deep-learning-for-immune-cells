# Week 18 (5) meeting minutes - Tue 11th February 2020

- I showed how my visualisation tool could be used to look at why outliers might be happening in clusters. It can be a bit slow if hovering over big clusters because of the computations happening in the background but it is useful.

- I now have a tool to run evaluations directly on my laptop, and surprisingly the visusalisation has yielded good clusters, which did not happen on Colab (I couldn't give a reason why). However this means that it might be worth look into the data more.
  - high-dimensional visualisation techniques such as umap or t-sne are useful to see if there is any structure in the data
  - but this means we could run a classifier on it

- I could use a classifier to take the encoded version of the images as inputs, with labels, and see if it can predict their classes on a held out dataset
  - I could try this with the given unstimulated, ConA, OVA labels first
  - but this might be more interesting to use as a binary classifier of more/less interaction
  - I could use the overlap metrics for this and a certain threshold to decide whether or not there is more/less interaction
  - this could be useful for evaluating the effect of compounds

- Carol was happy with this idea as it could be a good way to quantify the effects of drugs

- Finally I said I wanted to look into if the images obtained from UNet could be used, for example to feed them into the autoencoder, or as a way to mask out the background, but I think the classifier is a more interesting route which I want to focus on first

- I then showed the sections I have written out for my Materials & Methods in my dissertation 
