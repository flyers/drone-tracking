// Copyright 2013 Naiyan Wang

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
using namespace std;

namespace caffe {


template <typename Dtype>
void StructureOutputLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Do nothing
}

template <typename Dtype>
__global__ void StructureOutputLayerBackward(const int nthreads, const Dtype* bottom_data,
   const Dtype* bottom_box, int width, int height, Dtype* bottom_diff, Dtype* losses){
  
  int id = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
  if (id < nthreads) {
    int n = id;
    Dtype loss = 0;
    for (int h = 0; h < height; ++h){
      for (int w = 0; w < width; ++w){
        int index = n * width * height + h * width + w;
        Dtype temp = bottom_data[index];
        if (temp > 20) temp = 20;
        if (temp < -20) temp = -20;
        Dtype prob = 1 / (1 + exp(-temp));
        if (w > bottom_box[n * 4] && w < bottom_box[n * 4 + 2] &&
          h > bottom_box[n * 4 + 1] && h < bottom_box[n * 4 + 3]){
            bottom_diff[index] = -1 + prob;
            loss -= log(prob);
        } else {
          bottom_diff[index] = prob;
          loss -= log(1 - prob);
        }
      }
    }
    losses[n] = loss;
  }
}

template <typename Dtype>
Dtype StructureOutputLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const Dtype* bottom_box = (*bottom)[1]->gpu_data();

  StructureOutputLayerBackward<Dtype><<<CAFFE_GET_BLOCKS2D(NUM_), CAFFE_CUDA_NUM_THREADS>>>(
    NUM_, bottom_data, bottom_box, WIDTH_, HEIGHT_, bottom_diff, losses.mutable_gpu_data());

  Dtype totalLoss;
  caffe_gpu_asum<Dtype> (NUM_, losses.gpu_data(), &totalLoss);
  return totalLoss / (*bottom)[0]->count();
  /*for (int i = 0; i < NUM_; ++i){
    LOG(INFO) << (*bottom)[2]->cpu_data()[i * 4] << " " <<
      (*bottom)[0]->cpu_diff()[i * CHANNELS_ * HEIGHT_ * WIDTH_];// << " " <<
      //(*top)[0]->cpu_data()[i * CHANNELS_ * HEIGHT_ * WIDTH_] << " ";
  }*/
}

INSTANTIATE_CLASS(StructureOutputLossLayer);

}  // namespace caffe
