# Hybrid-Spatial-Spectral-Autoencoder-Models-for-Lossy-Satellite-Image-Compression

## Table of Contents

1. [Introduction](#introduction)
2. [Repository Files Description](#repository-files-description)
3. [Usage](#usage)
4. [Contact](#contact)
5. [Reference](#Reference)
   
## 1 - Introduction

The challenge of handling vast amounts of high-resolution satellite imagery is driven by onboard memory and bandwidth limitations. As spatial and spectral resolutions increase, image compression, particularly deep-learning based methods, is essential to overcome these limitations. This paper presents hybrid autoencoder models that combine convolutional neural networks, long short-term memory networks, and attention mechanisms (AM) for spatial and spectral feature extraction. The proposed architectures, including sparse and variational autoencoder (AE) counterparts, form a comprehensive image compression framework with quantization, and various entropy coders are applied to the EuroSat dataset (RGB and multispectral). The experimental results show the models’ superiority over the JPEG family and recent state-of-the-art methods, achieving up to 3.3, 1.4, and 0.6% improvements in the peak signal-to-noise ratio, structural similarity index, and multiscale structural similarity index, respectively. Moreover, performance analysis in terms of computational complexity, processing time, and memory usage highlights the efficiency of the proposed models. A case study conducted on a real scene from the Sentinel-2 satellite further validates the compatibility of the proposed models with modern artificial intelligence chipsets.

## 2 - Repository Files Description
```
Hybrid-Spatial-Spectral-Autoencoder-Models-for-Lossy-Satellite-Image-Compression/models/
├── Hybrid Casade (CNN-LSTM) AE                              
├── Hybrid Parallel (One directional CNN-LSTM) AE                        
├── Hybrid Parallel (Two directional CNN-LSTM) AE            
└── Hybrid Parallel (CNN-Attention) AE             
```
#### 1- Hybrid Casade (CNN-LSTM) AE
The propsoed model architecture is presented for both RGB and Multispectral EuroSat dataset. It utilizes both CNNs and LSTMs within a cascading architecture to efficiently process image data by leveraging spatial and spectral feature extraction methods. The proposed hybrid (CNN-LSTM) model combines the Spatial feature extraction capability of CNNs with the shared output feature and sequential context representation of LSTMs (Temporal feature extraction) to extract SPATIAL features to create an effective latent compact representation. The convolution layer is responsible for extracting spatial features from the input image, The proposed CNN output is refined forward via a flatten layer, which converts all of the resulting multidimensional arrays into a single long continuous linear vector from pooled feature maps, the input of three layers of stacked LSTM cells, respectively. An LSTM cell comprises three gates- input, output, and forget. The sequential latent representation represents the compressed data

#### 2- Hybrid Parallel (One directional CNN-LSTM) AE
The proposed model has two main components: a forward network(encoder) and a Backward network (decoder). We focused especially on separating spectral-spatial feature extraction blocks, which form the core of the SSFE network. The spectral and spatial features are combined into a spatial-spectral feature representation. The outputs of these blocks are then (Feature Fusion) concatenated and fed into a Downsampling Stage. The propsoed model architecture is presented for both RGB and Multispectral EuroSat dataset. The proposed hybrid SSFE model merges one directional CNN as a spatial block and LSTM as a spectral block in parallel paths, in which the CNN path focuses on spatial feature extraction, whereas the LSTM path is dedicated to spectral feature extraction.

#### 3- Hybrid Parallel (Two directional CNN-LSTM) AE
The propsoed model architecture is presented for both RGB and Multispectral EuroSat dataset. CNNs are adept at extracting spatial features from RGB images where spectral details are less critical. In contrast, for multispectral image compression, a standard CNN may ignore vital spectral information that is essential to these types of data. To address the mentioned issue, we propose a two-directional CNN approach ( this method allows the convolutional kernel to independently extract spatial features along the two parallel pathways, in which spatial features are extracted from two different directions, and makes full use of the correlations between rows and between columns of each pixel.)  With the characteristics of the sliding window mechanism of the CNN, it’s possible to capture integrated spatial features by simply altering the movement direction of the kernel, given the relative nature of the image tensor arrangement and the movement of the convolution kernel, transposing the image tensor is adopted as an alternative approach.

#### 4- Hybrid Parallel (CNN-Attention) AE 
The propsoed model architecture is presented for both RGB and Multispectral EuroSat dataset. The proposed hybrid AE model combines CNNs as a spatial block, using the same architecture as the Hybrid Parallel (CNN-LSTM) SSFE Blocks, with CNNs and Attention as a spectral block. The CNN path focuses on spatial decorrelation and feature extraction, whereas CNNs with Attention path are dedicated to spectral decorrelation and feature extraction processes.


## 3 - Usage
To implement and verify these models, you need to specify the dataset, the model name, and the path to the model's weights.

### You can download the RGB Dataset from this link: 
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset?select=EuroSATallBands

### You can download the Multispectral Dataset from this link:
https://www.kaggle.com/datasets/waseemalastal/eurosat-rgb-dataset

## 4 - Contact

## Mohamed Ahmed Badr, Researcher at Avionics Engineering Department, Military Technical College, Cairo, Egypt, m.badr1086@gmail.com

## AhmedFathyElrewainy, Assistant Professor, Avionics Engineering Department, Military Technical College, Cairo, Egypt, ahmed.elrewainy@mtc.edu.eg

## Mohamed Abdelmoneim Taha Elshafey, Associate Professor, Head of Computer Engineering and Artificial Intelligence Department, Military Technical College, Cairo, Egypt, m.shafey@mtc.edu.eg ; mohamed.elshafey@ieee.org

## 5 - Reference

```
@article{doi:10.2514/1.I011445,
author = {Badr, Mohamed Ahmed and Elrewainy, Ahmed Fathy and Elshafey, Mohamed Abdelmoneim Taha},
title = {Hybrid Spatial–Spectral Autoencoder Models for Lossy Satellite Image Compression},
journal = {Journal of Aerospace Information Systems},
volume = {0},
number = {0},
pages = {1-22},
year = {0},
doi = {10.2514/1.I011445},
URL = {https://doi.org/10.2514/1.I011445}
}
```
https://arc.aiaa.org/doi/10.2514/1.I011445
