# InsightFace

PyTorch implementation of Additive Angular Margin Loss for Deep Face Recognition.
[paper](https://arxiv.org/pdf/1801.07698.pdf).
```
@article{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
journal={arXiv:1801.07698},
year={2018}
}
```
## Performance

- sgd with momentum
- margin-m = 0.6
- margin-s = 64.0
- batch size = 256
- input image is normalized with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

|Models|MegaFace|LFW|Download|
|---|---|---|---|
|SE-LResNet101E-IR|97.43%|99.80%|[Link](https://github.com/foamliu/InsightFace-v3/releases/download/v1.0/insight-face-v3.pt)|


## Dataset

Function|Dataset|
|---|---|
|Train|MS-Celeb-1M|
|Test|MegaFace|

### Introduction

MS-Celeb-1M dataset for training, 3,804,846 faces over 85,164 identities.


## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage

### Data wrangling
Extract images, scan them, to get bounding boxes and landmarks:
```bash
$ python extract.py
$ python pre_process.py
```

Image alignment:
1. Face detection(Retinaface mobilenet0.25).
2. Face alignment(similar transformation).
3. Central face selection.
4. Resize -> 112x112. 

Original | Aligned & Resized | Original | Aligned & Resized |
|---|---|---|---|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_img.jpg)|

### Train
```bash
$ python train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```

## Performance evaluation

### MegaFace
 
#### Introduction
 
[MegaFace](http://megaface.cs.washington.edu/) dataset includes 1,027,060 faces, 690,572 identities.
 
Challenge 1 is taken to test our model with 1 million distractors. 

![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/megaface_stats.png)
 
#### Download

1. Download MegaFace and FaceScrub Images
2. Download FaceScrub annotation files:
    - facescrub_actors.txt
    - facescrub_actresses.txt
3. Download Linux DevKit from [MagaFace WebSite](http://megaface.cs.washington.edu/) then extract to megaface folder:

```bash
$ tar -vxf linux-devkit.tar.gz
```

#### Face Alignment

1. Align Megaface images:

```bash
$ python3 align_megaface.py
```

2. Align FaceScrub images with annotations:

```bash
$ python3 align_facescrub.py
```

#### Evaluation

```bash
$ python3 megaface_eval.py
```

It does following things:
1. Generate features for FaceScrub and MegaFace.
2. Remove noises. 
<br>Note: we used the noises list proposed by InsightFace, at https://github.com/deepinsight/insightface.
3. Start MegaFace evaluation through devkit. 

#### Results

##### Curves

Draw curves with matlab script @ megaface/draw_curve.m. 

CMC|ROC|
|---|---|
|![image](https://github.com/foamliu/InsightFace-v3/raw/master/images/megaface_cmc.jpg)|![image](https://github.com/foamliu/InsightFace-v3/raw/master/images/megaface_roc.jpg)|
|![image](https://github.com/foamliu/InsightFace-v3/raw/master/images/megaface_cmc_2.jpg)|![image](https://github.com/foamliu/InsightFace-v3/raw/master/images/megaface_roc_2.jpg)|

##### Textual results
<pre>
Done matching! Score matrix size: 3379 972313
Saving to results/otherFiles/facescrub_megaface_0_1000000_1.bin
Loaded 3379 probes spanning 80 classes
Loading from results/otherFiles/facescrub_facescrub_0.bin
Probe score matrix size: 3379 3379
distractor score matrix size: 3379 972313
Done loading. Time to compute some stats!
Finding top distractors!
Done sorting distractor scores
Making gallery!
Done Making Gallery!
Allocating ranks (972393)

Rank 1: <b>0.974266</b>

</pre>
