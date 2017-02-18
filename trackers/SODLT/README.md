# Installation
Download [caffe source code](https://www.dropbox.com/s/t6xy5cpsml86wiv/caffe.zip?dl=0). Note this a modified version of the official caffe toolbox. Put the download caffe code in directory ./external.


Download [pretrained caffe model](https://www.dropbox.com/s/iq4pfizbdkdww23/caffe_objectness_train_iter_100000?dl=0). Put the pretrained model in directory ./external.


Follow the [official caffe installation guide](http://caffe.berkeleyvision.org/installation.html) to compile caffe and the matlab wrapper. Note that GPU (with memory > 2GB) is required.

# Set environment variables

### caffe
Modifly the path in file ./external/caffe/examples/objectness/imagenet_deploy_solver.prototxt as

```sh
train_net: "/your_caffe_path/caffe/examples/objectness/imagenet_deploy.prototxt"
```

For any problems, please contact sliay@cse.ust.hk.

# Demo
```sh
run_individual
```

# CVPR13 benchmark
The code run_SODLT.m is compatiable with the evaluation toolkit in the CVPR13 benchmark.

# About
This code is an implementation of our work "Transferring Rich Feature Hierarchies for Robust Visual Tracking", Naiyan Wang, Siyi Li, Abhinav Gupta, Dit-Yan Yeung

If you have any problems with the codes or find bugs in codes, please contact winsty@gmail.com or sliay@ust.hk.