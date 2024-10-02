This is the official code repository for 

# Confidence Guided Semi-supervised Learning in Land Cover Classification

**Wanli Ma, Oktay Karakus, Paul L. Rosin**

*Semi-supervised learning has been well developed to help reduce the cost of manual labelling by exploiting a large quantity of unlabelled data. Especially in the application of land cover classification, pixel-level manual labelling in large-scale imagery is labour-intensive, time-consuming and expensive. However, existing semi-supervised learning methods pay limited attention to the quality of pseudo-labels during training even though the quality of training data is one of the critical factors determining network performance. In order to fill this gap, we develop a \textit{confidence-guided semi-supervised learning} (CGSSL) approach to make use of high-confidence pseudo labels and reduce the negative effect of low-confidence ones for land cover classification. Meanwhile, the proposed semi-supervised learning approach uses multiple network architectures to increase the diversity of pseudo labels. The proposed semi-supervised learning approach significantly improves the performance of land cover classification compared to the classic semi-supervised learning methods and even outperforms fully supervised learning with a complete set of labelled imagery of the benchmark Potsdam land cover dataset.


# [Paper link](https://ieeexplore.ieee.org/abstract/document/10281770)

# Framework
![arch](https://github.com/oktaykarakus/ReSIF/blob/main/CGSSL/figures/CGSSL.png?raw=true)
Overall framework of the confidence guided semi-supervised learning (CGSSL) approach.

![arch](https://github.com/oktaykarakus/ReSIF/blob/main/CGSSL/figures/CGCE.png?raw=true)
The details of the Confidence-Guided Cross Entropy (CGCE) module.

![arch](https://github.com/oktaykarakus/ReSIF/blob/main/CGSSL/figures/visual_results.png?raw=true)
Visual Results of each method on Potsdam Dataset. Values in parentheses refer to percentage accuracy. $^\#$U-Net1 was trained with the whole 3456 labelled samples. $^*$U-Net2 was trained with 1728 labelled samples.

# Installation
The code has been test under python 3.7.11. To install the environment from Anaconda:
```
conda env create -r requirements.yml
```
# Quite Starts
Potsdam dataset can be download from the [link](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) We divided these data tiles into $512\times512$ patches, resulting in 3456 training samples and 2016 test samples.
 
 Data structure for train, test and validation:
 ```
  │ data/
  ├── mypotsdam/
  │   ├── ann_dir/
  │   │   ├── test
  │   │   │   ├── ......
  │   │   ├── 2downsample_train
  │   │   │   ├── ......
  │   ├── dsm_dir/
  │   │   ├── test
  │   │   │   ├── ......
  │   │   ├── train
  │   │   │   ├── ......
  │   │   ├── 2downsample_train
  │   │   │   ├── ......
  │   ├── img_dir/
  │   │   ├── test
  │   │   │   ├── ......
  │   │   ├── train
  │   │   │   ├── ......
  │   │   ├── 2downsample_train
  │   │   │   ├── ......
  ```
  2downsample_train includes the labelled dataset, while the train consists of unlabeled data. Validation data is randomly selected as 10% of the labelled dataset.
  
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
@INPROCEEDINGS{10281770,
  author={Ma, Wanli and Karakuş, Oktay and Rosin, Paul L.},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Confidence Guided Semi-Supervised Learning in Land Cover Classification}, 
  year={2023},
  volume={},
  number={},
  pages={5487-5490},
  keywords={Training;Performance evaluation;Supervised learning;Training data;Manuals;Semisupervised learning;Network architecture;Semi-supervised Learning;Land Cover Classification;Multi-modality;Confidence Guided Loss},
  doi={10.1109/IGARSS52108.2023.10281770}}

```

