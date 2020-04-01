# Deep learning for analysing immune cell interactions

<object data="dissertation/figures/system_diagram.pdf" type="application/pdf">
    <embed src="dissertation/figures/system_diagram.pdf">
        <p>This browser does not support PDFs. Find the image in dissertation/figures/system_diagram.pdf</p>
    </embed>
</object>

### Background

_Deep learning for analysing immune cell interactions_ was my research project for my final year at the University of Glasgow. It was developed over a period of ~7 months from September 2019 to April 2020. This project is fully documented in `/dissertation/`.

This repository includes an implementation of a convolutional autoencoder and a deep regression model that were developed to research whether we could quality/quantify interaction from images of immune cells (T cells and dendritic cells).

### Project aims

* Can we find an underlying structure in the images of immune cells under different experimental conditions?
* Can a deep learning model be trained to 'quantify interaction' from an image of immune cells?

### Dataset specifics

I tried to find cluster of images of T cells and dendritic cells around their level of stimulation: _Unstimulated_, stimulation by _OVA peptide_, and stimulation by _ConA_.

Our 'quantity of interaction' was amount of overlap between the T cells and dendritic cells.

### Project structure

```
.
├── data            <-- find sample data, evaluation data, and notebooks
├── dissertation    <-- latex files and figures for generating my dissertation
├── meetings        <-- meeting minutes and powerpoint presentations
├── presentation    <-- final presentation slides
├── src             <-- code of the project
└── status_report   <-- status reports submitted after summer and before christmas
```

Instructions for running the code can be found in [src/README.md](src/README.md). If you want to explore the code you can look at [src/manual.md](src/manual.md).
