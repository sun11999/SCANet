# SCANet

Code and data for "Evolution and Duration of Mars’ Hesperian Ocean in Utopia Planitia".

## System Requirements

The codes are tested on Linux operating systems. (Linux: Ubuntu 22.04)

### Hardware Requirements
A standard computer with a RTX3090 GPU.

### Software Requirements
Python 3.9

Pytorch 1.8.0

### Python Dependencies
```
numpy
openpyxl
```

## Installation Guide
Download folder 'code'.

Prepare the testing enviroment. 

The codes should take approximately 10 seconds to install with vignettes on a recommended computer.

## Running
```
python predict.py
```
The code will take less than 6 seconds running on example SWIR spectra and predict the Mn concentration.
