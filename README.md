# Deep Compressed Sensing Network

Single-pixel imaging (SPI) enables the use of advanced detector technologies to provide a potentially low-cost solution for sensing beyond the visible spectrum and has received increasing attentions recently. However, when it comes to sub-Nyquist sampling, the spectrum truncation and spectrum discretization effects significantly challenge the traditional SPI pipeline due to the lack of sufficient sparsity. In this work, a deep compressive sensing (CS) framework is built to conduct image reconstructions in classical SPIs, where a novel compression network is proposed to enable collaborative sparsity in discretized feature space while remaining excellent coherence with the sensing basis as per CS conditions. To alleviate the underlying limitations in an end-to-end supervised training, e.g., the network typically needs to be re-trained as the basis patterns, sampling ratios, etc. change, the network is trained in an unsupervised fashion with no sensing physics involved.
## Overview



## Datasets and Video Examples



### Trainset



### Testsets



## Running Times



## Results


## SPI system schematic
<img src="https://github.com/Jeremy-jia2021/deep-compressed-sensing/blob/master/imgs/4.jpg" heigth=350>


## Architecture
<img src="https://github.com/Jeremy-jia2021/deep-compressed-sensing/blob/master/imgs/1.jpg" heigth=350>


## Code User Guide

### Colab example



### Dependencies

The code runs on Python +3.6. You can create a conda environment with all the dependecies by running
```
conda env create -f requirements.yml -n <env_name>
```

NOTE: the code was updated to support a newer version of the DALI library. For the original version of the algorithm which supported pytorch=1.0.0 and nvidia-dali==0.10.0 you can see this [release](https://github.com/m-tassano/fastdvdnet/releases/tag/v0.1)

### Testing



### Training


## Acknowledgement
I would like to offer a special thanks to Dr. Huijuan Zhao for being an excellent supervisor during my PhD journey. All the way through my academic life, I have been continuously inspired by her kindness, dedication, and love.
## ABOUT

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved. This file is offered as-is,
without any warranty.

* Author    : Mengyu (Jeremy) Jia
* Copyright : (C) 2022 Mengyu Jia
* Licence   : GPL v3+, see GPLv3.txt

