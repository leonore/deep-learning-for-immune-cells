# Week 22 (9) meeting minutes - Wed 11th March 2020

* I started the meeting by saying that I have almost finished my evaluation, but have been running into Google Colab GPU usage restrictions so it is taking a bit longer than I thought

* From the evaluation, here is what I learnt
  * t-SNE and UMAP do not yield useful clusters
  * there does not seem to be a structure to be learnt in different drug conditions; this might come from the small sub-image size that causes confusions
  * On the other hand, regression works well and the model can predict an interaction value from an image well

* In the two-category dataset, some odd clusters are appearing: if I get the time I will look into using the concentration variation labels to see if the clusters make sense from that

* I then went over some dissertation writing issues. The main issue I have been having at this point is knowing what to put in background or whether to keep it for another section.
  * Carol suggested to keep t-SNE, UMAP explanations in the background
  * but for specific choices like PReLU over ReLU, it is not really a "contributing factor" to the research and can just be mentioned in implementation.

* As the presentation will be carried out on March 23rd next week I should have a presentation ready so I can run over it for Carol and Hannah.
