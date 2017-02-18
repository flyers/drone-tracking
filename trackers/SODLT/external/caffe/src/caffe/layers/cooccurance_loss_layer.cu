 // Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/common.hpp"
using std::max;

namespace caffe {

template <typename Dtype>
__global__ void Im2Vec(const int n, const Dtype* in, Dtype* out, int channels, int height, int width){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n){
    int w = index % width;
    index /= width;
    int h = index % height;
    index /= height;
    int n = index;
    for (int i = 0; i < channels; ++i){
      out[(n * width * height + h * width + w) * channels + i] =
        in[n * channels * width * height + i * width * height + h * width + w];
    }
  }
}

template <typename Dtype>
__global__ void Vec2Im(const int n, const Dtype* in, Dtype* out, int channels, int height, int width){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n){
    int w = index % width;
    index /= width;
    int h = index % height;
    index /= height;
    int n = index;
    Dtype sum = 0;
    for (int i = 0; i < channels; ++i){
      sum += in[(n * width * height + h * width + w) * channels + i] * 
        in[(n * width * height + h * width + w) * channels + i];
    }
    sum = sqrt(sum);
    for (int i = 0; i < channels; ++i){
      out[n * channels * width * height + i * width * height + h * width + w] =
        in[(n * width * height + h * width + w) * channels + i] / sum;
    }
  }
}

template <typename Dtype>
__global__ void NeighborDiff(const int n, const Dtype* in, Dtype* diff, int channels,
  int height, int width, int offsetW, int offsetH, Dtype* losses){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int id = index;
  if (index < n){
    int w = index % width;
    index /= width;
    int h = index % height;
    index /= height;
    int n = index;
    int tw = w + offsetW;
    int th = h + offsetH;
    int loop = id * channels;
    Dtype loss = 0;
    if (tw >= 0 && th >= 0 && tw < width && th < height){
      int posId = n * width * height + th * width + tw;
      int posLoop = posId * channels;
      for (int i = 0; i < channels; ++i){
        Dtype temp = in[loop] - in[posLoop++];
        diff[loop++] += temp;
        loss += temp * temp;
      }
    }
    losses[index] = loss;
  }
}

template <typename Dtype>
__global__ void NegDiff(const int n, const Dtype* pos, const Dtype* neg, Dtype* diff,
  int channels, Dtype* losses, Dtype multiplier){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n){
    int posLoop = index * channels;
    int negLoop = 0;
    Dtype loss = 0;
    for (int i = 0; i < channels; ++i){
      Dtype temp = (pos[posLoop] - neg[negLoop++]) * multiplier;
      diff[posLoop++] -= temp;
      loss += temp * temp;
    }
    losses[index] = loss;
  }
}

template <typename Dtype>
__global__ void VectorSum(const int n, const Dtype* in, Dtype* out, const int num,
  const int channels) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n){
    out[index] = 0;
    for (int i = 0; i < num; ++i){
      out[index] += in[i * channels + index] / num;
    }
  }
}

template <typename Dtype>
inline int CooccuranceLossLayer<Dtype>::getNegId(double x, int i){
  int negId = x * (NUM_ - 1) * WIDTH_ * HEIGHT_;
  // This is used to skip the positions in the image self.
  if (negId >= i * WIDTH_ * HEIGHT_){
    negId += WIDTH_ * HEIGHT_;
  }
  return negId;
}

template <typename Dtype>
void CooccuranceLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Do nothing
}

template <typename Dtype>
Dtype CooccuranceLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype* col_data_ = col_buffer_.mutable_gpu_data();
  Dtype* col_diff_ = col_buffer_.mutable_gpu_diff();
  Dtype* temp_neg_data_ = temp_neg_.mutable_gpu_data();
  Dtype* temp_neg_diff_ = temp_neg_.mutable_gpu_diff();
  Dtype* pos_avg_data_ = pos_avg_.mutable_gpu_data();
  Dtype* neg_avg_data_ = neg_avg_.mutable_gpu_data();
  Dtype* losses_data_ = losses.mutable_gpu_data();

  Dtype* bottom_data_ = (*bottom)[0]->mutable_gpu_data();
  col_buffer_.Clear(true);
  Im2Vec<Dtype><<<CAFFE_GET_BLOCKS(NUM_ * WIDTH_ * HEIGHT_), CAFFE_CUDA_NUM_THREADS>>>
    (NUM_ * WIDTH_ * HEIGHT_, bottom_data_, col_data_, CHANNELS_, HEIGHT_, WIDTH_);
  CUDA_POST_KERNEL_CHECK;

  //Dtype totalLoss = 0;
  // For horizontal neighbors
  NeighborDiff<Dtype><<<CAFFE_GET_BLOCKS(NUM_ * WIDTH_ * HEIGHT_), CAFFE_CUDA_NUM_THREADS>>>
    (NUM_ * WIDTH_ * HEIGHT_, col_data_, col_diff_, CHANNELS_, HEIGHT_, WIDTH_,
    1, 0, losses_data_);
  NeighborDiff<Dtype><<<CAFFE_GET_BLOCKS(NUM_ * WIDTH_ * HEIGHT_), CAFFE_CUDA_NUM_THREADS>>>
    (NUM_ * WIDTH_ * HEIGHT_, col_data_, col_diff_, CHANNELS_, HEIGHT_, WIDTH_,
    -1, 0, losses_data_);
  //caffe_gpu_asum<Dtype> (NUM_ * WIDTH_ * HEIGHT_, losses.gpu_data(), &totalLoss);
  CUDA_POST_KERNEL_CHECK;
  // We need to use double here because of the precision of float may not be enough.
  // -1 because we need to exclude the image itself.
  vector<double> rand_ind_(NUM_ * (NEG_SAMPLE_NUM));
  caffe_vRngUniform<double>(rand_ind_.size(), &(rand_ind_[0]), 0, 1);

  // We add a barier here to make full use of both CPU and GPU.
  cudaDeviceSynchronize();
  // For Negative samples
  for (int i = 0; i < NUM_; ++i){
    int randNum = NEG_SAMPLE_NUM;
    temp_neg_.Clear(true);
    // Prepare the negative data
    for (int j = 0; j < randNum; ++j){
      int negId = getNegId(rand_ind_[i * randNum + j], i);
      cudaMemcpy(temp_neg_.mutable_gpu_data() + temp_neg_.offset(j), 
        col_buffer_.gpu_data() + col_buffer_.offset(negId), sizeof(Dtype) * CHANNELS_,
        cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    VectorSum<<<CAFFE_GET_BLOCKS(CHANNELS_), CAFFE_CUDA_NUM_THREADS>>>
      (CHANNELS_, col_data_ + col_buffer_.offset(i * WIDTH_ * HEIGHT_),
      pos_avg_data_, WIDTH_ * HEIGHT_, CHANNELS_);

    VectorSum<<<CAFFE_GET_BLOCKS(CHANNELS_), CAFFE_CUDA_NUM_THREADS>>>
      (CHANNELS_, temp_neg_data_, neg_avg_data_, randNum, CHANNELS_);

    NegDiff<Dtype><<<CAFFE_GET_BLOCKS(WIDTH_ * HEIGHT_), CAFFE_CUDA_NUM_THREADS>>>
      (WIDTH_ * HEIGHT_, col_data_ + col_buffer_.offset(i * WIDTH_ * HEIGHT_),
        neg_avg_data_, col_diff_ + col_buffer_.offset(i * WIDTH_ * HEIGHT_), CHANNELS_,
        losses_data_, Dtype(1.0));
    NegDiff<Dtype><<<CAFFE_GET_BLOCKS(randNum), CAFFE_CUDA_NUM_THREADS>>>
      (randNum, temp_neg_data_, pos_avg_data_, temp_neg_diff_, CHANNELS_, losses_data_,
        Dtype(WIDTH_ * HEIGHT_) / randNum);

    for (int j = 0; j < randNum; ++j){
      int negId = getNegId(rand_ind_[i * randNum + j], i);
      caffe_gpu_axpy<Dtype>(CHANNELS_, 1.0, temp_neg_.gpu_diff() + temp_neg_.offset(j),
        col_diff_ + col_buffer_.offset(negId), j % MULTIPLY_BATCH);
    }
  }
  for (int i = 0; i < NUM_; ++i){
    for (int iter = 0; iter < 10; ++iter){
       LOG(INFO) << col_buffer_.diff_at(i * WIDTH_ * HEIGHT_ + 3 * WIDTH_ + 3, iter, 0, 0);
    }
    LOG(INFO) << "====";
  }
  Dtype* bottom_diff_ = (*bottom)[0]->mutable_gpu_diff();
  Vec2Im<Dtype><<<CAFFE_GET_BLOCKS(NUM_ * WIDTH_ * HEIGHT_), CAFFE_CUDA_NUM_THREADS>>>
    (NUM_ * WIDTH_ * HEIGHT_, col_diff_, bottom_diff_, CHANNELS_, HEIGHT_, WIDTH_);
  //return loss / NUM_;
  return Dtype(0.0);
}

INSTANTIATE_CLASS(CooccuranceLossLayer);
}
