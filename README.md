# Deep Compressed Sensing Network - Official PyTorch Implementation
The paper is published
This repository provides the official PyTorch implementation of the following paper:
> **Single pixel imaging via unsupervised deep compressive sensing with collaborative sparsity in discretized feature space**<br>
> Mengyu Jia<sup>1</sup>, Lequan Yu <sup>2</sup>, Wenxing Bai<sup>1</sup>, Pengfei Zhang<sup>1</sup>, Limin Zhang<sup>1</sup>, Wei Wang<sup>3</sup>, Feng Gao<sup>1</sup> <br/>
> <sup>1</sup> College of Precision Instrument and Optoelectronics Engineering, Tianjin University, Tianjin 300072, China <br>
> <sup>2</sup> Department of Statistics and Actuarial Science, The University of Hong Kong, Hong Kong SAR, China <br/>
> <sup>3</sup> Department of Radiation Oncology, Tianjin Medical University Cancer Institute and Hospital, National Clinical Research Center for Cancer, Tianjin’s Clinical Research Center for Cancer, Key Laboratory of Cancer Prevention and Therapy, Tianjin, China, 300060 <br>
>
> **Abstract:**  Single-pixel imaging (SPI) enables the use of advanced detector technologies to provide a potentially low-cost solution for sensing beyond the visible spectrum and has received increasing attentions recently. However, when it comes to sub-Nyquist sampling, the spectrum truncation and spectrum discretization effects significantly challenge the traditional SPI pipeline due to the lack of sufficient sparsity. In this work, a deep compressive sensing (CS) framework is built to conduct image reconstructions in classical SPIs, where a novel compression network is proposed to enable collaborative sparsity in discretized feature space while remaining excellent coherence with the sensing basis as per CS conditions. To alleviate the underlying limitations in an end-to-end supervised training, e.g., the network typically needs to be re-trained as the basis patterns, sampling ratios, etc. change, the network is trained in an unsupervised fashion with no sensing physics involved.


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

