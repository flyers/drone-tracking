// Copyright 2014 Naiyan Wang
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <climits>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using namespace std;
using std::max;

namespace caffe {

template <typename Dtype>
void CooccuranceLossLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Cooccurance loss Layer takes one blobs as input.";
  CHECK_EQ(top->size(), 0) << "Cooccurance loss Layer takes no output.";
  NUM_ = bottom[0]->num();
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  MULTIPLY_BATCH = this->layer_param_.multiply_batch();
  NEG_SAMPLE_NUM = this->layer_param_.sampling_num();
  CHECK_EQ(NUM_ % MULTIPLY_BATCH, 0) << "Batch size should be dividable by multiply_batch.";
  col_buffer_.Reshape(NUM_ * HEIGHT_ * WIDTH_, CHANNELS_, 1, 1);
  pos_avg_.Reshape(1, CHANNELS_, 1, 1);
  neg_avg_.Reshape(1, CHANNELS_, 1, 1);
  temp_neg_.Reshape(NEG_SAMPLE_NUM, CHANNELS_, 1, 1);
  losses.Reshape(NUM_, 1, WIDTH_, HEIGHT_);
}

template <typename Dtype>
Dtype CooccuranceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CooccuranceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top){
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(CooccuranceLossLayer);
} // namespace caffe
