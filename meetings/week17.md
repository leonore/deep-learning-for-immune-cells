# Week 17 (4) meeting minutes - Wed 5th February 2019

- I outlined the progress I made on assembling all needed datasets together, looking at the dissertation, and starting to convert jupyter notebooks to Python files

- I voiced my worries about the amount of deep learning methods in the dissertation, and the fact that image segmentation isn't deep, but I do feed it back into the autoencoder and use it with UNet. I said I could look into making a classifier with a dense layer out of the autoencoder.
  - Carol said I could look into it, provided I had time, but that I had plenty of things to talk about in the dissertation.
  - (As soon as it's 2 layers it's considered "deep")

- I then showed the visualisation tool I have made, which Carol liked and thought would be good for the presentation
  - It is also my tool for seeing how outlier might go wrong

- I told Carol about going to the immunology labs on Friday, where I was able to take some notes on the processing done by the proprietary software.
  - On the other hand I wasn't able to take pictures of the masks obtained by the software, so I can't evaluate it this way.
  - I mentioned that a lot of research mentions doing flat field correction in image preprocessing. This is something that is offered by the proprietary software, but does not seem to have been applied to the images used as they have backgrounds which seem impactful as previously observed.
  - I tried different tactics to see if the backgrounds could be evened out, but unsuccessfully.
    - But this would be in line with the dataset with the masked out background working better than the plain one.
    - So I will be using that as my "evening background" method
