# Week 16 (3) - Fri 31st January 2020 @ Immunology

This is the notes I took while looking at the IN Cell Analyzer software.

#### Flat field correction
- It is unclear whether or not this is has been included in the preprocessing pipeline, but it is something the analyser software offers.

### Protocol for image analysis

##### t-cells
- Preprocessing --> unable to know what is done
- Object segmentation (kernel size 5, sensitivity 17)
- Postprocessing
  - Sieve (binary) (area > 40)
  - Watershed clump breaking
- Measures
  - Cell count
  - Area size

##### dendritic cells
- Preprocessing --> unable to know what is done
- Object segmentation (kernel size 15, sensitivity 16)
- Postprocessing
  - Sieve (binary) (area > 18)

##### Overlap
- Preprocessing
- Intensity segmentation
  - Low threshold 1.0, upper threshold 4095.0
- Measures
  - Area overlap

### Methods

- **Object segmentation**
  - Uses top hat transformation
  - *Kernel size*: size of the window from which to make the computations, e.g. 3x3 uses 8 other cells for 1
  - *Sensitivity*: level of an object's brightness relative to its background

- **Sieve (binary)**
  - Filters the segmented object according to a certain threshold (area > x)

- **Watershed clump breaking**
  - Separates multiple objects clumped together

- **Count**
  - Number of targets contained within the outline 
