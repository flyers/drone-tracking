 #!/usr/bin/env sh

GLOG_logtostderr=1 ./build/examples/test_net.bin examples/rotate_deeper/imagenet_val.prototxt /data/nwangab/imagenet_snapshot/caffe_imagenet_train_iter_410000 1250 VAL  GPU 2
