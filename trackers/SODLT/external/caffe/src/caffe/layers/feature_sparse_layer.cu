// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeatureSparseLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  lambda_ = this->layer_param_.lambda();
};

template <typename Dtype>
void FeatureSparseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
Dtype FeatureSparseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FeatureSparseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // Do nothing
}

template <typename Dtype>
__global__ void FeatureSparseBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype lambda_) {
  int index = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x;
  if (index < n) {
    out_diff[index] = in_diff[index] + lambda_ * in_data[index];
  }
}

template <typename Dtype>
Dtype FeatureSparseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    FeatureSparseBackward<Dtype><<<CAFFE_GET_BLOCKS2D(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, lambda_);
    // for (int i = 0; i < 10; ++i){
    //   LOG(INFO) << (*bottom)[0]->data_at(0, 0, 0, i * 10) << " "
    //     << (top)[0]->diff_at(0, 0, 0, i * 10);
    // }
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(FeatureSparseLayer);

}  // namespace caffe
