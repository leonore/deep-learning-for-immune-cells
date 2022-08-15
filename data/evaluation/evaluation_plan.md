# Evaluation plan

## Artifacts to evaluate

* Autoencoder
* Clustering
* Image segmentation
* Classification

## Evaluation methods

### Autoencoder

1. Plot loss
  * Compare loss values to first tries with full image, without sliding window
2. **Show reconstructed image** 
  * Visual satisfaction
  * Progress compared to relu feature maps + reconstructed without sliding window

### Clustering

* For each dataset:
  1. Look at how different the images look under human visualisation
  2. Run UMAP (faster) or t-sne (more well researched)
  3. Labels can tell us if clustering is satisfactory
  4. Look at outliers if outlier visualisation function works
* If clusters are obtained: it worked!

### Image segmentation

#### As a technique
1. Compare speed of K-means vs. thresholding
2. Compare quality of K-means vs. thresholding (is it satisfactory?)
3. Evaluate thresholding tactic with another b&w dataset e.g. Fashion MNIST

#### For calculating overlap metrics

1. Calculate overlap metrics
2. Record to a file
3. Plot results in difference to target results
