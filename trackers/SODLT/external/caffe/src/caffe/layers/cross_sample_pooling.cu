// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void CrossSamplePoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Cross Sample Pooling Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Cross Sample Pooling Layer takes a single blob as output.";

  // Get the pooling parameters
  NUM_ = bottom[0]->num();
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  (*top)[0]->Reshape(1, CHANNELS_, HEIGHT_, WIDTH_);

  max_id_.reset(new Blob<int>(1, CHANNELS_, HEIGHT_, WIDTH_));
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, Dtype* top_data) {
  int index = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x;
  if (index < nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height);
    Dtype aveval = 0;
    for (int i = 0; i < num; ++i){
      aveval += bottom_data[((i * channels + c) * height + h) * width + w];
    }
    top_data[index] = aveval / num;
  }  // (if index < nthreads)
}

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, Dtype* top_data, int* max_id_) {
  int index = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x;
  if (index < nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    for (int i = 0; i < num; ++i){
      Dtype temp = bottom_data[((i * channels + c) * height + h) * width + w];
      if (temp > maxval){
        maxval = temp;
        maxidx = i;
      }
    }
    top_data[index] = maxval;
    max_id_[index] = maxidx;
  }  // (if index < nthreads)
}

template <typename Dtype>
void CrossSamplePoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Adaptive Batch size.
  NUM_ = min(bottom[0]->num(), this->GetNet()->GetBatchSize());
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  switch (this->layer_param_.pool()) {
    case LayerParameter_PoolMethod_MAX:
      MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS2D(CHANNELS_ * WIDTH_ * HEIGHT_), CAFFE_CUDA_NUM_THREADS>>>(
          CHANNELS_ * WIDTH_ * HEIGHT_, bottom_data, NUM_, CHANNELS_,
          HEIGHT_, WIDTH_, top_data, max_id_->mutable_gpu_data());
      break;
    case LayerParameter_PoolMethod_AVE:
      AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS2D(CHANNELS_ * WIDTH_ * HEIGHT_), CAFFE_CUDA_NUM_THREADS>>>(
          CHANNELS_ * WIDTH_ * HEIGHT_, bottom_data, NUM_, CHANNELS_,
          HEIGHT_, WIDTH_, top_data);
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, Dtype* bottom_diff, const int* max_id_) {
  int index = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x;
  if (index < nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height);
    for (int i = 0; i < num; ++i){
      bottom_diff[((i * channels + c) * height + h) * width + w] = 
        (max_id_[index] == i ? top_diff[c * height * width + h * width + w] : 0);
    }
  }  // (if index < nthreads)
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, Dtype* bottom_diff) {
  int index = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x;
  if (index < nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height);
    Dtype grad = top_diff[c * height * width + h * width + w];
    for (int i = 0; i < num; ++i){
      bottom_diff[((i * channels + c) * height + h) * width + w] = grad / num;
    }
  }  // (if index < nthreads)
}

template <typename Dtype>
Dtype CrossSamplePoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return Dtype(0.);
  }
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_data();
  const Dtype* top_diff = top[0]->gpu_data();
  switch (this->layer_param_.pool()) {
    case LayerParameter_PoolMethod_MAX:
      MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS2D(CHANNELS_ * WIDTH_ * HEIGHT_), CAFFE_CUDA_NUM_THREADS>>>(
          CHANNELS_ * WIDTH_ * HEIGHT_, top_diff, NUM_, CHANNELS_,
          HEIGHT_, WIDTH_, bottom_diff, max_id_->gpu_data());
      break;
    case LayerParameter_PoolMethod_AVE:
      AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS2D(CHANNELS_ * WIDTH_ * HEIGHT_), CAFFE_CUDA_NUM_THREADS>>>(
          CHANNELS_ * WIDTH_ * HEIGHT_, top_diff, NUM_, CHANNELS_,
          HEIGHT_, WIDTH_, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  return Dtype(0.0);
}

template <typename Dtype>
Dtype CrossSamplePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
  return Dtype(0.0);
}

template <typename Dtype>
void CrossSamplePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(CrossSamplePoolingLayer);

} // end namespace caffe