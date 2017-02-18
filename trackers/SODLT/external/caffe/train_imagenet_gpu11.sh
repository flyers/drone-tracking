#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/examples/train_net.bin ./examples/gpu11_multires_rotate_deeper/imagenet_solver.prototxt #/data/nwangab/imagenet_snapshot/caffe_imagenet_train_iter_700000.solverstate
