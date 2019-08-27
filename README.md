## A Gobal-local Embedding Module for Fashion Landmark Detection

Source code for the paper 'A Global-local Embedding Module for Fashion Landmark Detection'

Authors : [Sumin Lee](https://sites.google.com/view/suminlee/), Sungchan Oh, Chanho Jung, Changick Kim

Accepted to ICCV 2019 Workshop [Computer Vision for Fashion, Art, and Design](https://sites.google.com/view/cvcreative/home?authuser=0)


### Requirements
- Python 3
- Pytorch >= 0.4.1
- torchvision

### Quick Start

1. Download the datasets
* Deepfashion [[download](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)]
* FLD [[download](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)]

2. Train
```
python train.py --root [root_directory] --dataset [dataset_option] --glem [glem_option]
```
'root' and 'dataset' options are necessary.

3. Evaluate
```
# You can run the file only for evaluation
python train.py --root [root_directory] --dataset [dataset_option] --glem [glem_option] --evaluate True
```

--------------

### Abstract

Detecting fashion landmarks is a fundamental technique for visual clothing analysis. Due to the large variation and non-rigid deformation of clothes, localizing fashion landmarks suffers from large spatial variances across poses, scales, and styles. Therefore, understanding contextual knowledge of clothes is required for accurate landmark detection. To that end, in this paper, we propose a fashion landmark detection network with a global-local embedding module. The global-local embedding module is based on a non-local operation for capturing long-range dependencies and a subsequent convolution operation for adopting local neighborhood relations. With this processing, the network can consider both global and local contextual knowledge for a clothing image. We demonstrate that our proposed method has an excellent ability to learn advanced deep feature representations for fashion landmark detection. Experimental results on two benchmark datasets show that the proposed network outperforms the state-of-the-art methods.

![architecture](./img/architecture.jpg)

--------------




We refered the code in [[site](https://github.com/fdjingyuan/Deep-Fashion-Analysis-ECCV2018)].

### Citation
