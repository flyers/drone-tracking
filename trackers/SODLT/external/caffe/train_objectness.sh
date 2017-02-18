#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/examples/train_net.bin ./examples/objectness/imagenet_solver.prototxt /media/windisk/imagenet_snapshot/caffe_objectness_train_iter_100000.solverstate
