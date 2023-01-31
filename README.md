# Tantalum Project

This repository contains all progress obtained in 6 months in the PhD program supported by Dr. Brandon Runnels and the grant offered by National Science Foundation.

## Objectives
Science of materials field of study have putting efforts to characterize the behavior of materials. Among many possible themes, this project aims to predict the behavior of grain boundaries after stress application using Multilayer Convolutional Neural Network (MCNN).

Since August 2022 my evotuion in this project can be described in the following macro timeline:

1. Void detection 
2. Voids Classification
3. Boundaries reconstruction 
4. Grain Coloring 
5. Boundaries Coloring 
6. Train and test dataset construction 

## Content
Folders in this project:
### Data
All original images that will work as a input file must be in this folder.
This folder also contains a txt file with the same name of the image. This file contains all information about each boundarie [More info...]()

### ml_sets
All images in this folder are already processed. The file name is defined as: original_file_name + tile position + number_of_voids + function

| Input File | Position [0-15] | Number of voids [0 - 100] | Function | Output File name |
|------------|-----------------|---------------------------|----------|------------------|
| 1_001      | 0               | 3                         | void     | 1_001_0_3void    |
| 1_001      | 0               | 3                         | remake   | 1_001_0_3void    |
| 1_001      | 0               | 3                         | proof    | 1_001_0_3proof   |

**Void**: Original image with voids.

**Remake**: Image after void reconstruction

**Proof**: Original image to compare with void and remake and help to identify void position

### output

This Folder contains the dataframe result of Void reconstruction. All information is stored in a pickle file to simplify the reprocessing step.



### source

This Folder contains the scripts that develop all the pre-processing part [More info...](source/README.md)

    

## Requirements
See [Requirements](requirements.txt)

