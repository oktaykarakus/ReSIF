This is the official code repository for 

# AMM-FuseNet: Attention-Based Multi-Modal Image Fusion Network for Land Cover Mapping

**Wanli Ma, Oktay Karakus, Paul L. Rosin**

*Abstract: Land cover mapping provides spatial information on the physical properties of the Earth's surface for various classes of wetlands, artificial surface and constructions, vineyards, water bodies, etc. Having reliable information on land cover is crucial to developing solutions to a variety of environmental problems, such as the destruction of important wetlands/forests, and loss of fish and wildlife habitats. This has made land cover mapping become one of the most widespread applications in remote sensing computational imaging. However, due to the differences between modalities in terms of resolutions, content, and sensors, integrating complementary information that multi-modal remote sensing imagery exhibits into a robust and accurate system still remains challenging, and classical segmentation approaches generally do not give satisfactory results for land cover mapping. In this paper, we propose a novel dynamic deep network architecture, **AMM-FuseNet** that promotes the use of multi-modal remote sensing images for the purpose of land cover mapping. The proposed network exploits the hybrid approach of the channel attention mechanism and densely connected atrous spatial pyramid pooling (DenseASPP). In the experimental analysis, in order  to verify the validity of the proposed method, we test AMM-FuseNet with three datasets whilst comparing it to the six state-of-the-art models of DeepLabV3+, PSPNet, UNet, SegNet, DenseASPP, and DANet. In addition, we  demonstrate the capability of AMM-FuseNet under minimal training supervision (reduced number of training samples) compared to the state of the art, achieving less accuracy loss, even for the case with 1/20 of the training samples.  


[Paper link](https://www.mdpi.com/2072-4292/14/18/4458)

[DOI](https://doi.org/10.3390/rs14184458)

# Framework
![arch](https://github.com/oktaykarakus/ReSIF/blob/main/AMM-FuseNet/figures/AMM-FuseNet.png)
# Requirements
coming soon
# Installation
coming soon
# Quite Starts
 we show an example for one dataset, Hunan. other datasets' preparation has the same structure with Hunan. Hunan dataset can be download from the [link](https://github.com/LauraChow/HunanMultimodalDataset)
 
 Data structure for train, test and validation:
 ```
  │ data/
  ├── myhunan/
  │   ├── ann_dir/
  │   │   ├── test
  │   │   │   ├── ......
  │   │   ├── train
  │   │   │   ├── ......
  │   │   ├── val
  │   │   │   ├── ......
  │   ├── dem_dir/
  │   │   ├── test
  │   │   │   ├── ......
  │   │   ├── train
  │   │   │   ├── ......
  │   │   ├── val
  │   │   │   ├── ......
  │   ├── s1_dir/
  │   │   ├── test
  │   │   │   ├── ......
  │   │   ├── train
  │   │   │   ├── ......
  │   │   ├── val
  │   │   │   ├── ......
  │   ├── s2_dir/
  │   │   ├── test
  │   │   │   ├── ......
  │   │   ├── train
  │   │   │   ├── ......
  │   │   ├── val
  │   │   │   ├── ......

  ```
  
  To retrain the model:
  ```
  python train.py
  ```
  
  To test the model:
  ```
  python test.py
  ```
  
# Citation

```
@Article{AMM_FuseNet,
AUTHOR = {Ma, Wanli and Karakuş, Oktay and Rosin, Paul L.},
TITLE = {AMM-FuseNet: Attention-Based Multi-Modal Image Fusion Network for Land Cover Mapping},
JOURNAL = {Remote Sensing},
VOLUME = {14},
YEAR = {2022},
NUMBER = {18},
ARTICLE-NUMBER = {4458},
URL = {https://www.mdpi.com/2072-4292/14/18/4458},
ISSN = {2072-4292},
DOI = {10.3390/rs14184458}
}

```
