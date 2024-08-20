# Pursuing Effective and Efficient Knowledge Distillation with Low Memory under Data-Free Conditions
This repository is the official code for the paper "Pursuing Effective and Efficient Knowledge Distillation with Low Memory under Data-Free Conditions"
## Introduction
To reproduce our results, please download pre-trained teacher models from [Dropbox-Models (266 MB)](https://www.dropbox.com/sh/w8xehuk7debnka3/AABhoazFReE_5mMeyvb4iUWoa?dl=0) and extract them as `checkpoints/pretrained`.

## Dependencies

* Python 3.6
* PyTorch 1.2.0
* Dependencies in requirements.txt

### Installation
Install pytorch and other dependencies:

        pip install -r requirements.txt


### Training

To train wrn16_2 (student model) with wrn40_2 (teacher model) on CIFAR-100, run the following command:

    bash scripts/gapssg/cifar100_wrn40_2_wrn16_2.sh






