# Evaluation plan

## Artifacts to evaluate

* Autoencoder
* Clustering (t-sne/UMAP)
* Image segmentation

## Evaluation methods

### Autoencoder

* Plot loss
  * Have an element of comparison, like the CIFAR dataset
  * Compare to first tries with full image
* Show reconstructed image
  * Visual satisfaction
  * Progress

### Clustering

* For each dataset:
  * Look at how different the images look under human visualisation
  * Run UMAP (faster) or t-sne (more well researched)
  * Labels can tell us if clustering is satisfactory
  * Look at outliers if outlier visualisation function works
* If clusters are obtained: it worked!

### Image segmentation

#### As a technique
* Compare speed of K-means vs. thresholding
* Compare quality of K-means vs. thresholding (is it satisfactory?)
* Evaluate thresholding tactic with another b&w dataset e.g. Fashion MNIST

#### For calculating overlap metrics

* Calculate overlap metrics
* Record to a file
* Plot results in difference to target results
