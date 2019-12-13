# Level 4 Individual project - Deep learning for analysing immune cell interactions

### Project aims

* Using an autoencoder to learn a compressed representation of cell images in order to evaluate whether this helps with dimensionality reduction and clustering the cell images according to different drug stimulates
* Exploring different image segmentation techniques in order to find binary masks of cells and calculate the overlap in order to analyse whether cells do or do not interact together under different conditions

### Project structure

* `timelog.md` The time log for the project.
* `plan.md` A skeleton week-by-week plan for the project.
* `data/` data acquired during the project
  * `*_logs.md` Step-by-step logs of results obtained with different autoencoder and clustering parameters
  * `results/` contains the images of the different results obtained as above.
  * `raw/` is for the pre-processing cell images (locally)
  * `processed/` is for the post-processing cell images (locally)
* `src/` all the source code for the project:
  * the working code, in Python notebooks
  * helper functions, in separate Python files
* `status_report/` includes the summer report and the status report to be submitted in December
* `meetings/` records of the meetings had during the project, including PowerPoint presentations made
* `dissertation/` source for project dissertation
* `presentation/` future presentation

### Initial project description

The initiation of an immune response depends on the interaction strength between different types of immune cells. Dendritic cells act as pathogen detectors and form an interaction with T-helper cells; the strength of these interactions determines if and how the immune response subsequently develops.     

This project aims to apply deep learning to the analysis of microscope images of interacting dendritic cells and T-helper cells (supplied by biological researchers). Algorithms will be developed for segmentation of the different cell types and to perform subsequent analysis, e.g. changes in cell morphology under different experimental conditions.     

The strength of the interactions between the two cell types can be influenced by the application of drugs that either inhibit or enhance the interactions. Inhibitors (such as abatacept) can be useful in treating autoimmune diseases such as rheumatoid arthritis, whereas enhancement can be useful in immunotherapy for the treatment of certain cancers (e.g. with check-point inhibitors), or for developing vaccines against infectious disease.     

The algorithms developed in the project will be used to explore the effects of drugs on immune cell interactions.    

Technology: Python, Keras/Tensorflow
