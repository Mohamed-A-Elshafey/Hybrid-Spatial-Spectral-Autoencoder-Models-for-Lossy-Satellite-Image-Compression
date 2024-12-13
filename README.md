# Hybrid-Spatial-Spectral-Autoencoder-Models-for-Lossy-Satellite-Image-Compression

## Table of Contents

1. [Introduction](#introduction)
2. [Repository Files Description](#repository-files-description)
3. [Usage](#usage)
4. [Reference](#Reference)
   
## 1 - Introduction

 The challenge of handling vast amounts of high-resolution satellite imagery is driven by onboard memory and bandwidth limitations. As spatial and spectral resolutions increase, image compression, particularly deep-learning based methods, is essential to overcome these limitations. This paper presents hybrid autoencoder models that combine convolutional neural networks, long short-term memory networks, and attention mechanisms for spatial and spectral feature extraction. The proposed architectures, including sparse and variational autoencoder counterparts, form a comprehensive image compression framework with quantization, and various entropy coders are applied to the EuroSat dataset (RGB and multispectral). The experimental results show the models’ superiority over the JPEG family and recent state-of-the-art methods, achieving up to 3.3, 1.4, and 0.6% improvements in the peak signal-to-noise ratio, structural similarity index, and multiscale structural similarity index, respectively. Moreover, performance analysis in terms of computational complexity, processing time, and memory usage highlights the efficiency of the proposed models. A case study conducted on a real scene from the Sentinel-2 satellite further validates the compatibility of the proposed models with modern artificial intelligence chipsets.

## 2 - Repository Files Description
```
Hybrid-Spatial-Spectral-Autoencoder-Models-for-Lossy-Satellite-Image-Compression/models/
├── Hybrid Casade (CNN-LSTM) AE                               # Script to evaluate model accuracy.
├── Hybrid Parallel (CNN-Attention) AE                        # Helper functions.
├── Hybrid Parallel (One directional CNN-LSTM) AE             # Used to train the Teacher.
└── Hybrid Parallel (Two directional CNN-LSTM) AE             # Contains the architectures of the Projector and the regressor.
```
#### 1- Hybrid Casade (CNN-LSTM) AE

#### 2- Hybrid Parallel (CNN-Attention) AE

#### 3- Hybrid Parallel (One directional CNN-LSTM) AE

#### 4- Hybrid Parallel (Two directional CNN-LSTM) AE


## 3 - Usage
To implement and verify these models, you need to specify the dataset, the model name, and the path to the model's weights.
You can download the RGB Dataset from this link: 
https:..........................................
You can download the Multispectral Dataset from this link:
https:........................................................


## 4 - Reference
**Under review**
```

