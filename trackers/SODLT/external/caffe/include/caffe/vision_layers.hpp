// Copyright 2013 Yangqing Jia

#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <leveldb/db.h>
#include <pthread.h>
#include <cstring>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


// The neuron layer is a specific type of layers that just works on single
// celements.
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
};


template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
 public:
  explicit ReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};


template <typename Dtype>
class SigmoidLayer : public NeuronLayer<Dtype> {
 public:
  explicit SigmoidLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};


template <typename Dtype>
class BNLLLayer : public NeuronLayer<Dtype> {
 public:
  explicit BNLLLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};


template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype> {
 public:
  explicit DropoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  shared_ptr<SyncedMemory> rand_vec_;
  float threshold_;
  float scale_;
  unsigned int uint_thres_;
};

template <typename Dtype>
class ReshapeLayer : public NeuronLayer<Dtype> {
 public:
  explicit ReshapeLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int bottomNum;
  int bottomChannels;
  int bottomHeight;
  int bottomWidth;
  int topNum;
  int topChannels;
  int topHeight;
  int topWidth;
};

template <typename Dtype>
class CeilingLayer : public NeuronLayer<Dtype> {
 public:
  explicit CeilingLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Dtype threshold_;
};

template <typename Dtype>
class FeatureSparseLayer : public NeuronLayer<Dtype> {
 public:
  explicit FeatureSparseLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Dtype lambda_;
};

template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
 public:
  explicit FlattenLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int count_;
};


template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int M_;
  int K_;
  int N_;
  bool biasterm_;
  shared_ptr<SyncedMemory> bias_multiplier_;
};


template <typename Dtype>
class PaddingLayer : public Layer<Dtype> {
 public:
  explicit PaddingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  unsigned int PAD_;
  int NUM_;
  int CHANNEL_;
  int HEIGHT_IN_;
  int WIDTH_IN_;
  int HEIGHT_OUT_;
  int WIDTH_OUT_;
};


template <typename Dtype>
class LRNLayer : public Layer<Dtype> {
 public:
  explicit LRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // scale_ stores the intermediate summing results
  Blob<Dtype> scale_;
  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  int num_;
  int channels_;
  int height_;
  int width_;
};


template <typename Dtype>
class Im2colLayer : public Layer<Dtype> {
 public:
  explicit Im2colLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int KSIZE_;
  int STRIDE_;
  int CHANNELS_;
  int HEIGHT_;
  int WIDTH_;
  int PAD_;
};


template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int CHANNELS_;
  int HEIGHT_;
  int WIDTH_;

  Blob<int> KSIZE_;
  Blob<int> STRIDE_;
  Blob<int> POOLED_HEIGHT_;
  Blob<int> POOLED_WIDTH_;
  vector<shared_ptr<Blob<float> > > rand_idx_;
  vector<shared_ptr<Blob<int> > > max_idx_;
};


template <typename Dtype>
class ConvolutionLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  void RenormalizeFilter();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  Blob<Dtype> col_bob_;

  int KSIZE_;
  int STRIDE_;
  int NUM_;
  int CHANNELS_;
  int HEIGHT_;
  int WIDTH_;
  int NUM_OUTPUT_;
  int GROUP_;
  int PAD_;
  int MULTIPLY_BATCH;
  Blob<Dtype> col_buffer_;
  shared_ptr<SyncedMemory> bias_multiplier_;
  bool biasterm_;
  int M_;
  int K_;
  int N_;
};


template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
     const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  // sum_multiplier is just used to carry out sum using blas
  Blob<Dtype> sum_multiplier_;
  // scale is an intermediate blob to hold temporary results.
  Blob<Dtype> scale_;
};


template <typename Dtype>
class MultinomialLogisticLossLayer : public Layer<Dtype> {
 public:
  explicit MultinomialLogisticLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class InfogainLossLayer : public Layer<Dtype> {
 public:
  explicit InfogainLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), infogain_() {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> infogain_;
};


// SoftmaxWithLossLayer is a layer that implements softmax and then computes
// the loss - it is preferred over softmax + multinomiallogisticloss in the
// gradients. During testing this layer could be replaced by a softmax layer
// to generate probability outputs.
template <typename Dtype>
class SoftmaxWithLossLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxWithLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
     const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  // prob stores the output probability of the layer.
  Blob<Dtype> prob_;
  // Vector holders to call the underlying softmax layer forward and backward.
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
};


template <typename Dtype>
class EuclideanLossLayer : public Layer<Dtype> {
 public:
  explicit EuclideanLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), difference_() {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  Blob<Dtype> difference_;
};


template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
  explicit AccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // The accuracy layer should not be used to compute backward operations.
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
    return Dtype(0.);
  }
};

template <typename Dtype>
class UWConvolutionLayer : public Layer<Dtype> {
 public:
  explicit UWConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  void RenormalizeFilter();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  Blob<Dtype> col_bob_;

  int KSIZE_;
  int STRIDE_;
  int NUM_;
  int CHANNELS_;
  int HEIGHT_;
  int WIDTH_;
  int NUM_OUTPUT_;
  int GROUP_;
  int PAD_;
  int MULTIPLY_BATCH;
  Blob<Dtype> col_buffer_;
  shared_ptr<SyncedMemory> bias_multiplier_;
  bool biasterm_;
  int M_;
  int K_;
  int N_;
  int DELTA_;
};

template <typename Dtype>
class MaxoutLayer : public Layer<Dtype> {
 public:
  explicit MaxoutLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int M_;
  int K_;
  int N_;
  int K_PER_GROUP;
  bool biasterm_;
  shared_ptr<SyncedMemory> bias_multiplier_;

  // Buffer for temp results
  shared_ptr<SyncedMemory> buffer_out;

  // Store the activated index
  shared_ptr<SyncedMemory> out_index;  
};

template <typename Dtype>
class RankingLossLayer : public Layer<Dtype> {
 public:
  explicit RankingLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class L2HingeLossLayer : public Layer<Dtype> {
 public:
  explicit L2HingeLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Dtype C_;
};

template <typename Dtype>
class ConcatLayer : public Layer<Dtype> {
 public:
  explicit ConcatLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  Blob<Dtype> col_bob_;

  int COUNT_;
  int NUM_;
  int CHANNELS_;
  int HEIGHT_;
  int WIDTH_;
  int concat_dim_;
};

template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int count_;
};

template <typename Dtype>
class CooccuranceLossLayer : public Layer<Dtype> {
 public:
  explicit CooccuranceLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), softmaxloss_layer_(new SoftmaxWithLossLayer<Dtype>(param)) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int getNegId(double x, int i);

  shared_ptr<SoftmaxWithLossLayer<Dtype> > softmaxloss_layer_;
  // prob stores the output probability of the layer.
  Blob<Dtype> neg_avg_;
  Blob<Dtype> temp_neg_;
  Blob<Dtype> pos_avg_;
  Blob<Dtype> col_buffer_;
  Blob<Dtype> losses;
  int NUM_;
  int CHANNELS_;
  int WIDTH_;
  int HEIGHT_;
  int MULTIPLY_BATCH;
  int NEG_SAMPLE_NUM;
};

template <typename Dtype>
class StructureOutputLossLayer : public Layer<Dtype> {
 public:
  explicit StructureOutputLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  int NUM_;
  int HEIGHT_;
  int WIDTH_;
  Blob<Dtype> losses;
};


template <typename Dtype>
class CrossSamplePoolingLayer : public Layer<Dtype> {
 public:
  explicit CrossSamplePoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int NUM_;
  int CHANNELS_;
  int HEIGHT_;
  int WIDTH_;
  shared_ptr<Blob<int> > max_id_;
};

template <typename Dtype>
class MultiLabelLogisticLossLayer : public Layer<Dtype> {
 public:
  explicit MultiLabelLogisticLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class VOCLossLayer : public Layer<Dtype> {
 public:
  explicit VOCLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class MaskLayer : public Layer<Dtype> {
 public:
  explicit MaskLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int output_scale_;
  shared_ptr<Blob<Dtype> > targets_;
};

}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_

