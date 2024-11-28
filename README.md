# DeFI-Net

## Overview

Image inpainting, which is the task of filling in missing areas in an image, is a common image editing technique. Inpainting can be used to conceal or alter image contents in malicious manipulation of images, driving the need for research in image inpainting detection. Existing methods mostly rely on a basic encoder-decoder structure, which often results in a high number of false positives or misses the inpainted regions, especially when dealing with targets of varying semantics and scales. Additionally, the absence of an effective approach to capture boundary artifacts leads to less accurate edge localization. In this paper, we describe a new method for inpainting detection based on a Dense Feature Interaction Network (DeFI-Net). DeFI-Net uses a novel feature pyramid architecture to capture and amplify multi-scale representations across various stages, thereby improving the detection of image inpainting by better revealing feature-level interactions. Additionally, the network can adaptively direct the lower-level features, which carry edge and shape information, to refine the localization of manipulated regions while integrating the higher-level semantic features. Using DeFI-Net, we develop a method combining complementary representations to accurately identify inpainted areas.  Evaluation on five image inpainting datasets demonstrate the effectiveness of our approach, which achieves state-of-the-art performance in detecting inpainting across diverse models.

## Network Architecture



## Pre-trained models

The pre-trained weight of backbone and the checkpoint on IID dataset and Defacto dataset can be downloaded on the link: https://pan.baidu.com/s/1SLUazBaU_YWfioEsbKOIyQ?pwd=29sn

## Data preparation

We provide the source and link to the dataset used in our work as follows:

[**IID_dataset**](https://github.com/HighwayWu/InpaintingForensics)，**[DEFACTO](https://defactodataset.github.io/)**，[**AutoSplice**](https://github.com/shanface33/autosplice_dataset)，[**Korus**](https://www.pkorus.pl/downloads)，[**CocoGlide**](https://github.com/grip-unina/TruFor)，[**TGIF**](https://github.com/IDLabMedia/tgif-dataset)

## Citation

