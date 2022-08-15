# Week 20 (7) meeting minutes - Wed 26th February 2020

- I went over the visualisation plots and metrics I have developed for the regression model.

- I then explained that the reason why I suddenly got clusters with UMAP was because I was mistakenly passing in the image labels in the function, which made it a supervised dimensionality reduction technique instead of being unsupervised.
  - This has now been corrected but it means clusters don't seem to be found without the labels.
  - On the other hand this pushed me to develop a regression model, which is good.

- I explained the final tweaks I made to the autoencoder and regression model.
  - Both have ReduceLROnPlateau and EarlyStopping callbacks and are now trained for 20 epochs (usually stop earlier)
  - The autoencoder compressed representation is now at about 1000 data points, which is a 100x reduced from the original image.
    - I also tried reducing it by 200x but the reconstruction was more satisfactory for 1000x
  - The regression model includes dropout to make it more robust
    - Showed the before:after visualisation plots for the improvement.

- I have started writing materials & methods in my dissertation.
  - I hope to have some draft done for next week that I can send to Carol.
  - Hannah will have to look over the biological explanations to make sure that they are correct.

- I then explained what I was going to evaluate.
  - There will be some systematic evaluations, but for some other things e.g. UNet, thresholding vs K-Means I will probably run it in a Jupyter notebook for any visualisation.

- This week will be a lot of code cleaning/refactoring and then running evaluations.
  - I want to have a config file
  - It would be good to have a sample dataset to show an example to be able to run from the repository.
    - Hannah should be able to provide images that were obtained by students
    - Otherwise I suggested skewing/rotating the images just to have a noisy dataset, it might not obtain results but things can still be ran.

    
