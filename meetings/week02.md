# Week 2 meeting minutes - 30 September 2019

* I went over what I had done over the past week (setup and readings)
* I asked Hannah to clarify some things about the cell images
    * We went over the spreadsheet describing the images: each number corresponds to a different compound ID number. For each interaction, we have a brightfield image, a "TexasRed" image (red colouring, dendritic cells) and a "FITC" image (green colouring, t-cells)
* As we don't have immediate access to masks of the images, I will try and use autoencoders to try and get a compressed representation of the images
    * I can then apply a clustering algorithm to the compressed representations to see if it can find when there is interactions, and when there is not
* Depending on how well the project goes, we would also want to use the representations of the different cells to try and calculate some differences/similarities (e.g. size, granularity)
* We also went over some formalities: Github (issues will be on Github, if I find that does not work I will make a Trello board), computer cluster access
