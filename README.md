# Pre-ResNet Generator
A simple python script to generate Caffe prototxt of pre-activation ResNet on the ImageNet (ILSVRC) dataset. See [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) and [fb's torch implementation](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua).

## Usage

1. Set the `CAFFE_ROOT` (line 8) properly.
2. `python2 net_def.py --depth 200`

Depth can be 50, 101, 152, or 200. Will output `resnet{depth}_trainval.txt` respectively by default.

## Note

1. The script only serves as a network scaffold. More data augmentation schemes should be added to the data layers to achieve good performance.

2. Some pre-activation BatchNorm layers could be further made inplace to save memory. Will double check the correctness of doing so.