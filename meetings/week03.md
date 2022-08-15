# Week 3 meeting minutes - 8 October 2019

* I went over what I have done in the past week, which was mostly getting to grips with reading the TIFF images to arrays. I also setup an autoencoder to make sure it ran with a small input.
* I asked about what resolution we should aim for for the images. I should trial and error some sizes to see how well it performs (MNIST 28*28 is too small, too much detail is lost)
* I then asked some questions about what kind of prediction task we were making. I have been struggling with conceptualising what kind of output we are actually aiming for for this project so I wanted to make sure I was put on the right track not to do any work that could not be used.
* I should work with the fluorescent green and red images, and run them through the autoencoder. I can then see how much the green and red images overlap (calculate it) and feed that into a clustering algorithm to see how it changes with the different drugs.
    * i.e. Run the autoencoder on all the data you have, calculate the overlap, then run it through a clustering algorithm
    * If that goes well, then future work could be done on looking at how cell morphology changes depending on the drugs.
    * (The brightfield images are hard to work with: it's hard to differentiate the t-cells from the dendritic cells)
* Now that I have a better idea of the project, I'm going to (try to) make a plan for the semester, and email it
* I also said I would have a look at the dissertation template, make sure it builds, and write down any background information, research, etc. that is useful to me as I go along to make the process of dissertation writing easier in the future
* We don't have a meeting next week, as both Carol and Hannah are unavailable, but I will email them a detailed PowerPoint of what I have done.
