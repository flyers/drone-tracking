 #!/usr/bin/env sh

GLOG_logtostderr=1 ./build/examples/test_net.bin examples/gpu11_unshared_weight_large_lr/imagenet_val.prototxt /data/nwangab/imagenet_snapshot/caffe_imagenet_train_iter_800000 1250 VAL  GPU 0
