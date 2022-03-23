# Deep Compressed Sensing Network - Official PyTorch Implementation

This repository provides the official PyTorch implementation of the following paper:
> **Single pixel imaging via unsupervised deep compressive sensing with collaborative sparsity in discretized feature space**<br>
> Mengyu Jia<sup>1</sup>, Lequan Yu <sup>2</sup>, Wenxing Bai<sup>1</sup>, Pengfei Zhang<sup>1</sup>, Limin Zhang<sup>1</sup>, Wei Wang<sup>3,+</sup>, Feng Gao<sup>1,+</sup> <br/>
>
> <sup>1</sup> College of Precision Instrument and Optoelectronics Engineering, Tianjin University, Tianjin 300072, China <br>
> <sup>2</sup> Department of Statistics and Actuarial Science, The University of Hong Kong, Hong Kong SAR, China <br/>
> <sup>3</sup> Department of Radiation Oncology, Tianjin Medical University Cancer Institute and Hospital, National Clinical Research Center for Cancer, Tianjin’s Clinical Research Center for Cancer, Key Laboratory of Cancer Prevention and Therapy, Tianjin, China, 300060 <br>
> 
> **Abstract:**  _Single-pixel imaging (SPI) enables the use of advanced detector technologies to provide a potentially low-cost solution for sensing beyond the visible spectrum and has received increasing attentions recently. However, when it comes to sub-Nyquist sampling, the spectrum truncation and spectrum discretization effects significantly challenge the traditional SPI pipeline due to the lack of sufficient sparsity. In this work, a deep compressive sensing (CS) framework is built to conduct image reconstructions in classical SPIs, where a novel compression network is proposed to enable collaborative sparsity in discretized feature space while remaining excellent coherence with the sensing basis as per CS conditions. To alleviate the underlying limitations in an end-to-end supervised training, e.g., the network typically needs to be re-trained as the basis patterns, sampling ratios, etc. change, the network is trained in an unsupervised fashion with no sensing physics involved._

## SPI system schematic

<img src="https://github.com/Jeremy-jia2021/deep-compressive-sensing/blob/master/imgs/4.jpg" width="600" height="300"/>


## Network Architecture

<img src="https://github.com/Jeremy-jia2021/deep-compressive-sensing/blob/master/imgs/1.jpg" width="600" height="300"/>


## Datasets
The training datasets used the hyperspectral data from [ICVL](http://icvl.cs.bgu.ac.il/hyperspectral-imaging/). Given our end point applications (fluorescence imaging), only one-channel data were used for training. <br>
The testing on natural images used the data from [Linnaeous](http://localhost:6001/#images) datasets. <br>
The in-vivo fluorescence imaging data can be downloaded [here](https://pan.baidu.com/s/1EiF5YkjjWnM7tHEG711l7g). (PWD: 1234) <br>

## Training networks
To train the network, run the training script below.
```
python train.py -opt options/train.yml
```

## Test networks
To test on simulated SPI measurements, run the script below<br>
```
python main_sim.py -opt options/test.yml
```

To test on experimental SPI measurements, run the script below<br>
```
python main_exp.py -opt options/test.yml
```

Reconstructions from the traditional CS can be performed by running <br>
```
python main_exp_wavelet.py -opt options/test.yml 
```
or
```
python main_sim_wavelet.py -opt options/test.yml
```
## Acknowledgement
I would like to offer a special thanks to Dr. Huijuan Zhao for being an excellent supervisor during my PhD journey. All the way through my academic life, I have been continuously inspired by her kindness, dedication, and love.

## Fundings
National Natural Science Foundation of China (62175183, 61575140, 62075156, 81871393, 81872472); The Science&Technology Development Fund of Tianjin Education Commission for Higher Education (2018KJ231).
## ABOUT
Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved. This file is offered as-is,
without any warranty.

* Author    : Mengyu (Jeremy) Jia (jiamengyu2004@126.com)
* Copyright : (C) 2022 Mengyu Jia
* Licence   : GPL v3+, see GPLv3.txt

