#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/examples/train_net.bin ./examples/multires_rotate_deeper/imagenet_solver.prototxt /media/windisk/imagenet_snapshot/caffe_imagenet_train_iter_720000.solverstate
