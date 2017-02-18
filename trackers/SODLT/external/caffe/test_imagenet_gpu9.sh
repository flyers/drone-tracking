 #!/usr/bin/env sh

GLOG_logtostderr=1 ./build/examples/test_net.bin examples/gpu9_zeiler/imagenet_val.prototxt /data/nwangab/imagenet_snapshot/caffe_imagenet_train_iter_520000 500  GPU
