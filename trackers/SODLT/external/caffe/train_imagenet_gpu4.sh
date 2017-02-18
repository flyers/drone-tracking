#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/examples/train_net.bin ./examples/rotate_deeper/imagenet_solver.prototxt /data/nwangab/imagenet_snapshot/caffe_imagenet_train_iter_410000.solverstate
