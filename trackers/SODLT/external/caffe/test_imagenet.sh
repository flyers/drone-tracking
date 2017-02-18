 #!/usr/bin/env sh

GLOG_logtostderr=1 ./build/examples/test_net.bin examples/multires_rotate_deeper/imagenet_val.prototxt /media/windisk/imagenet_snapshot/caffe_imagenet_train_iter_800000 250 VAL GPU 0
