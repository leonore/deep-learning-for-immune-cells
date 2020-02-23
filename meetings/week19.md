# Week 19 (6) meeting minutes - Wed 19th February 2020

- Before the meeting Hannah had sent over a research article that might be of interest to us, so we started by discussing the paper.
  - Paper URL: https://elifesciences.org/articles/06994
  - I can use this paper in the background/motivation part of my dissertation as it explores some of the immunology concepts that are of interest
    - Figures:    
      (1) naive, normal antigen.   
      There is still some communication between the cells.         
      (2) 200nm = best immunisation response. cells are recognising themselves    
      Reduced speed of movement.   
      At 72 hours still reaction.

- I went over the system diagram I drew, showing how I have been using the weights obtained from the autoencoder as the foundation for a classifier/deep regression model.
  - Those weights were trained on the balanced dataset (same amounts of images in each category).
  - I first combined the overlap labels I have from calculation with IoU to see if it could find the category of the image (unstimulated, ova, cona, faulty).
  - It achieved an accuracy of around 52% on the test dataset.
  - However the categories are rigid and it would be harder to extend to something else.
  - So I developed a deep regression model instead that takes a picture and tries to predict the overlap value.
  - I didn't develop a binary metric because choosing a threshold value to say, values above x are interaction, values under x are non interaction, seemed reductive and overly simplified.
    - There's also the issue that I talked about with Hannah that cells might seem to be interacting in 2D, but they could just be sitting on top of each other (which seems to happen often with unstimulated cells)

- I went over how I calculate my overlap metric, which is just intersection over union obtained from the binary version of the images. However this is calculated per sub-image, so to kind of see how well the overlap replicates the true values calculated by software I sum each sub-image overlap value together and compare the distributions for that with the true values.
  - For faulty images I give them the label obtained from the main image, but give them an overlap value of 0.

- I mentioned that I wasn't sure how UNet fit into all this.
  - It works, it was easy to follow a tutorial for it.
  - Carol said I could mention it for possible future work.
    - Could use it for feature extraction.
    - Could use it to analyse cell structure and cell morphology changes.

- We talked about the presentation a bit, because if I have time I might look into using tensorboard for cluster visualisations for it as it looks a bit nicer than the matplotlib tool I developed. 
