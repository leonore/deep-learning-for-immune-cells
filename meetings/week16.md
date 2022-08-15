# Week 16 (3) meeting minutes - Tue 28th January 2020

- First I showed my final autoencoder code and explained my decisions for choosing bigger filter sizes, and extra layer, strides in the last layer, and PReLU vs. ReLU.
  - This model is still quite simple and it is for the simplicity of explanation as well as the track record of simple models performing well.
  - PReLU and striding performed slightly better than without.
  - Bigger filter sizes gets slightly more quality but the extra pooling layer still gives smaller dimensions for the encoded layer.

- I then showed my final processing code, highlighting that I give a separate label for "faulty" images to see if the algorithm can detect accordingly, and so that it does not impact clustering performance.

- Then I showed that OpenCV performs much better than sklearn for k-means, with up to 12x speedup for the exact same result. Thresholding, on the other hand, does not get the exact same result with some slight differences but still gets a massive speedup of about 28x.
  - Both can be evaluated this way with the clustering.

- I collated all the research metrics result I need for the three datasets I chose to use.
  - I justified my decisions in terms of how they can help to see how clustering + autoencoder performs.
  - I still need to build two datasets but have worries about the memory usage this takes. I have some ideas on how to bypass this with Numpy datatypes, but in case this does not, I might have to halve the dataset. However that should be okay as they will still be slightly bigger than the DMSO dataset I currently have.

- I also now have my evaluation plan written out which I can follow as a methodology for my evaluation. It also gives me a structure for my dissertation. 

- I still need to meet Hannah in order to see how the GE software they currently use for analysing cells performs and which methods they use.

- I mentioned I am ahead of schedule. Carol and Hannah suggested that I start writing my dissertation now not only to relieve some stress, but also to help me guide my thinking. I might realise I need to look into something else as I am writing, and it is best to notice this now than later.

- In the coming week I will be working on getting the datasets finished, as well as starting to write up the dissertation as Carol and Hannah said it would be a good idea.
  - Moreover, if Hannah can organise for me to come in to look at the software, I will write up on that.
