// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CeilingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  threshold_ = this->layer_param_.threshold();
};

template <typename Dtype>
void CeilingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
Dtype CeilingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
__global__ void CeilingForward(const int n, const Dtype* in, Dtype* out, Dtype threshold_) {
  int index = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x;
  if (index < n) {
    out[index] = in[index] < threshold_ ? in[index] : threshold_;
  }
}

template <typename Dtype>
__global__ void CeilingBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype threshold_) {
  int index = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x;
  if (index < n) {
    out_diff[index] = in_diff[index] * (in_data[index] < threshold_);
  }
}

template <typename Dtype>
void CeilingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  CeilingForward<Dtype><<<CAFFE_GET_BLOCKS2D(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, threshold_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
Dtype CeilingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    CeilingBackward<Dtype><<<CAFFE_GET_BLOCKS2D(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, threshold_);
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(CeilingLayer);

}  // namespace caffe
