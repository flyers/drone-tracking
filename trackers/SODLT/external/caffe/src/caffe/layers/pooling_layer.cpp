// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void PoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "PoolingLayer takes a single blob as input.";
  LOG(INFO) << "The pooling layer generates " << top->size() << " scale outputs.";
  CHECK_EQ(top->size(), this->layer_param_.poolsize_size());
  CHECK_EQ(top->size(), this->layer_param_.poolstride_size());

  // Get the pooling parameters
  KSIZE_.Reshape(top->size(), 1, 1, 1);
  STRIDE_.Reshape(top->size(), 1, 1, 1);
  POOLED_HEIGHT_.Reshape(top->size(), 1, 1, 1);
  POOLED_WIDTH_.Reshape(top->size(), 1, 1, 1);
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  for (int i = 0; i < top->size(); ++i){
    KSIZE_.mutable_cpu_data()[i] = this->layer_param_.poolsize(i);
    STRIDE_.mutable_cpu_data()[i] = this->layer_param_.poolstride(i);
    int ksize = this->layer_param_.poolsize(i);
    int stride = this->layer_param_.poolstride(i);
    int pooled_height = ceil(static_cast<float>(HEIGHT_ - ksize) / stride) + 1;
    int pooled_width = ceil(static_cast<float>(WIDTH_ - ksize) / stride) + 1;
    POOLED_HEIGHT_.mutable_cpu_data()[i] = pooled_height;
    POOLED_WIDTH_.mutable_cpu_data()[i] = pooled_width;
    (*top)[i]->Reshape(bottom[0]->num(), CHANNELS_, pooled_height,
      pooled_width);

    // Initialize the intermediate variables for different pooling method
    if (this->layer_param_.pool() == LayerParameter_PoolMethod_STOCHASTIC) {
      shared_ptr<Blob<float> > pointer(new Blob<float>(bottom[0]->num(), CHANNELS_, pooled_height,
        pooled_width));
      rand_idx_.push_back(pointer);
    }
    if (this->layer_param_.pool() == LayerParameter_PoolMethod_MAX) {
      shared_ptr<Blob<int> > pointer(new Blob<int>(bottom[0]->num(), CHANNELS_, pooled_height,
        pooled_width));
      max_idx_.push_back(pointer);
    }
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  LOG(ERROR) << "WARN: You are running the old version of CPU pooling.";
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  /*int top_count = (*top)[0]->count();
  switch (this->layer_param_.pool()) {
  case LayerParameter_PoolMethod_MAX:
    // Initialize
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = -FLT_MAX;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < CHANNELS_; ++c) {
        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
          for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
            int hstart = ph * STRIDE_;
            int wstart = pw * STRIDE_;
            int hend = min(hstart + KSIZE_, HEIGHT_);
            int wend = min(wstart + KSIZE_, WIDTH_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * POOLED_WIDTH_ + pw] =
                  max(top_data[ph * POOLED_WIDTH_ + pw],
                      bottom_data[h * WIDTH_ + w]);
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  case LayerParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < CHANNELS_; ++c) {
        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
          for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
            int hstart = ph * STRIDE_;
            int wstart = pw * STRIDE_;
            int hend = min(hstart + KSIZE_, HEIGHT_);
            int wend = min(wstart + KSIZE_, WIDTH_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * POOLED_WIDTH_ + pw] +=
                    bottom_data[h * WIDTH_ + w];
              }
            }
            top_data[ph * POOLED_WIDTH_ + pw] /=
                (hend - hstart) * (wend - wstart);
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  case LayerParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }*/
}

template <typename Dtype>
Dtype PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  LOG(FATAL) << "CPU version is not implemented";
  /*if (!propagate_down) {
    return Dtype(0.);
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  memset(bottom_diff, 0, (*bottom)[0]->count() * sizeof(Dtype));
  switch (this->layer_param_.pool()) {
  case LayerParameter_PoolMethod_MAX:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < CHANNELS_; ++c) {
        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
          for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
            int hstart = ph * STRIDE_;
            int wstart = pw * STRIDE_;
            int hend = min(hstart + KSIZE_, HEIGHT_);
            int wend = min(wstart + KSIZE_, WIDTH_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * WIDTH_ + w] +=
                    top_diff[ph * POOLED_WIDTH_ + pw] *
                    (bottom_data[h * WIDTH_ + w] ==
                        top_data[ph * POOLED_WIDTH_ + pw]);
              }
            }
          }
        }
        // offset
        bottom_data += (*bottom)[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case LayerParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < CHANNELS_; ++c) {
        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
          for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
            int hstart = ph * STRIDE_;
            int wstart = pw * STRIDE_;
            int hend = min(hstart + KSIZE_, HEIGHT_);
            int wend = min(wstart + KSIZE_, WIDTH_);
            int poolsize = (hend - hstart) * (wend - wstart);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * WIDTH_ + w] +=
                  top_diff[ph * POOLED_WIDTH_ + pw] / poolsize;
              }
            }
          }
        }
        // offset
        bottom_data += (*bottom)[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case LayerParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }*/
  return Dtype(0.);
}


INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
