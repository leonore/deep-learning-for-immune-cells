# Level 4 Individual project - Deep learning for analysing immune cell interactions

### Guidance for myself - to remember

I need:   
* a timelog, updated regularly in the `timelog.md` format;
* all source under version control;
* data well organised and with appropriate ethical approval (for human subject data);

Here's an overview of the structure as it stands:

* `timelog.md` The time log for the project.
* `plan.md` A skeleton week-by-week plan for the project.
* `questions.md` Any questions to ask to Hannah or Carol in next meeting
* `data/` data acquired during the project
* `src/` source code for your project
* `status_report/` the status report submitted in December
* `summer/summer.md` contains a record of what was done over the summer prior to the project starting
* `meetings/` Records of the meetings you have during the project.
* `dissertation/` source for project dissertation
* `presentation/` future presentation

* Make sure you add a `.gitignore` or similar for your VCS for the tools you are using!

### Initial project description

The initiation of an immune response depends on the interaction strength between different types of immune cells. Dendritic cells act as pathogen detectors and form an interaction with T-helper cells; the strength of these interactions determines if and how the immune response subsequently develops.     

This project aims to apply deep learning to the analysis of microscope images of interacting dendritic cells and T-helper cells (supplied by biological researchers). Algorithms will be developed for segmentation of the different cell types and to perform subsequent analysis, e.g. changes in cell morphology under different experimental conditions.     

The strength of the interactions between the two cell types can be influenced by the application of drugs that either inhibit or enhance the interactions. Inhibitors (such as abatacept) can be useful in treating autoimmune diseases such as rheumatoid arthritis, whereas enhancement can be useful in immunotherapy for the treatment of certain cancers (e.g. with check-point inhibitors), or for developing vaccines against infectious disease.     

The algorithms developed in the project will be used to explore the effects of drugs on immune cell interactions.    
      
Technology: Python, Keras/Tensorflow
