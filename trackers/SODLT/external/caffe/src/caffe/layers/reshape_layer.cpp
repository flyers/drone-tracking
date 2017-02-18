// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  // Set up the cache for random number generation
  bottomNum = bottom[0]->num();
  bottomChannels = bottom[0]->channels();
  bottomHeight = bottom[0]->height();
  bottomWidth = bottom[0]->width();
  topNum = this->layer_param_.num();
  topChannels = this->layer_param_.channels();
  topHeight = this->layer_param_.height();
  topWidth = this->layer_param_.width();
  CHECK_EQ(bottomNum * bottomChannels * bottomHeight * bottomWidth,
    topNum * topChannels * topHeight * topWidth);
  (*top)[0]->Reshape(topNum, topChannels, topHeight, topWidth);
};

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->ReshapeNoErase(topNum, topChannels, topHeight, topWidth);
}

template <typename Dtype>
Dtype ReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 20; ++i){
    LOG(INFO) << (*bottom)[0]->data_at(0, i, 4, 4) << " " << (*bottom)[0]->diff_at(0, i, 4, 4);
  }
  (*bottom)[0]->ReshapeNoErase(bottomNum, bottomChannels, bottomHeight, bottomWidth);
  return 0;
}

INSTANTIATE_CLASS(ReshapeLayer);

}  // namespace caffe
