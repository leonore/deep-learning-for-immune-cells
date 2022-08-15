# Research on data normalisation

## Why normalise?
- CNNs learn by continually adding gradient error vectors multiplied by a learning rate computed from backpropagation to various weight matrices throughout the network as training examples are passed through
- if we don't scale our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ proportionally speaking from one another
- (source: https://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn, need something more reputable)

## Which method to use?
- minmax normalisation keeps values within a range [0,1], same scale
  - but is sensitive to outliers

- standardisation (z-score normalisation): using arithmetic mean and standard deviation
  - less sensitive outliers
  - if the input scores are not gaussian the distribution is not gaussian at output

- https://datascience.stackexchange.com/questions/54296/should-input-images-be-normalized-to-1-to-1-or-0-to-1

- https://stats.stackexchange.com/questions/330559/why-is-tanh-almost-always-better-than-sigmoid-as-an-activation-function/369538

- http://cs231n.github.io/neural-networks-2/#datapre

- https://stackoverflow.com/questions/33610825/normalization-in-image-processing

- https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current

- https://arxiv.org/pdf/1611.04251.pdf

- https://www.sciencedirect.com/science/article/pii/S1361841517301135

- http://cs231n.github.io/neural-networks-2/#datapre

- https://cs.nyu.edu/media/publications/sermanet_pierre.pdf

- http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf 
