# Week 5 meeting minutes - 21 Oct 2019

- I showed the work I've been doing over the past two weeks tuning the autoencoder.
- I mentioned that I can get a fairly good decompressed representation out, but I'm not losing any dimensionality
   - Carol said that this doesn't necessarily matter.
   - We could be trying to cluster the images without using the autoencoder at all.
   - However, the reason why we're doing this is that it's not guaranteed we could actually get something meaningful out of this, hence the different representation, from which we should be able to get meaningful information. I suggested trying to run my code on different inputs/outputs of the autoencoder as well once the clustering algorithm is set up to see how it performs.
- I raised the fact that the images aren't labelled at all right now, which won't be very useful for clustering or highlighting in which conditions the cells react most. I should label the cells according to the stimulant, which is the most relevant label at the moment.
- Overlap calculations should be done two by two, for the t-cell and dendritic cells in the same conditions.
- Carol suggested looking at a 200x200 middle cropping of the cell images, rather than resizing them.
- Ideally, if I have the time, it would be good to create separate scripts to run this programme in order for it to be used by researchers.
- I have my training day for access to the computer cluster next week, but I will also have a look at uploading the resized pictures to my Google Drive in order to be able to use them with Google Colab and exploit its GPU power.
