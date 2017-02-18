// Copyright 2013 Yangqing Jia

#include <vector>
#include <fstream>
#include <cublas_v2.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;

namespace caffe {

template <typename Dtype>
void UWConvolutionLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Conv Layer takes a single blob as output.";
  KSIZE_ = this->layer_param_.kernelsize();
  STRIDE_ = this->layer_param_.stride();
  GROUP_ = this->layer_param_.group();
  MULTIPLY_BATCH = this->layer_param_.multiply_batch();
  NUM_ = bottom[0]->num();
  CHECK_EQ(NUM_ % MULTIPLY_BATCH, 0);
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  NUM_OUTPUT_ = this->layer_param_.num_output();
  PAD_ = this->layer_param_.pad();
  CHECK_GT(NUM_OUTPUT_, 0);
  CHECK_EQ(CHANNELS_ % GROUP_, 0);
  CHECK_EQ(NUM_ % MULTIPLY_BATCH, 0);
  int height_out = (HEIGHT_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  int width_out = (WIDTH_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  col_buffer_.Reshape(MULTIPLY_BATCH, CHANNELS_ * KSIZE_ * KSIZE_, 
      height_out , width_out);
  // Set the parameters
  CHECK_EQ(NUM_OUTPUT_ % GROUP_, 0)
      << "Number of output should be multiples of group.";
  biasterm_ = this->layer_param_.biasterm();
  // Figure out the dimensions for individual gemms.
  M_ = NUM_OUTPUT_ / GROUP_;
  K_ = CHANNELS_ * KSIZE_ * KSIZE_ / GROUP_;
  N_ = height_out * width_out;
  (*top)[0]->Reshape(bottom[0]->num(), NUM_OUTPUT_, height_out, width_out);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization"; } else {
    if (biasterm_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(
        new Blob<Dtype>(NUM_OUTPUT_*N_, CHANNELS_ / GROUP_, KSIZE_, KSIZE_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(
        GetFiller<Dtype>(this->layer_param_.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // If necessary, intiialize and fill the bias term
    if (biasterm_) {
      LOG(INFO) << "Init bias term for conv layer";
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, NUM_OUTPUT_));
      shared_ptr<Filler<Dtype> > bias_filler(
          GetFiller<Dtype>(this->layer_param_.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Set up the bias filler
  if (biasterm_) {
    bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
};

template <typename Dtype>
void UWConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  LOG(FATAL)<<"Forward_cpu not implemented.";
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //Dtype* bottom_data = botton[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* top_data_gpu = (*top)[0]->mutable_gpu_data();
  Dtype* top_label = (*top)[1]->mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_data_gpu = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * 1;
  shared_ptr<Blob<Dtype> > bottom_data_clone;
  bottom_data_clone.reset(new Blob<Dtype>(1, 1, CHANNELS_ * KSIZE_ * KSIZE_, N_));
  Dtype* bottom_clone = bottom_data_clone->mutable_gpu_data();
  shared_ptr<Blob<Dtype> > top_data_clone;
  top_data_clone.reset(new Blob<Dtype>(1, 1, N_, M_*GROUP_ ));
  Dtype* top_clone = top_data_clone->mutable_gpu_data();
  for (int n = 0; n < NUM_; ++n) {
    // First, im2col
    im2col_cpu(bottom_data+bottom[0]->offset(n), CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, PAD_, STRIDE_, col_data);
    caffe_matrix_transpose<Dtype>(col_data_gpu, bottom_clone, K_*GROUP_, N_);
    // Second, innerproduct with groups
    for(int d=0;d<N_;d++){
    	for (int g = 0; g < GROUP_; ++g) {
      		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, 1, K_,
        		(Dtype)1., weight + d*M_*K_*GROUP_ + weight_offset * g,  bottom_data_clone->mutable_cpu_data() + d*K_*GROUP_ + K_ * g,
        		(Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * (d*GROUP_+g));
	}
    }
    CUDA_CHECK(cudaMemcpy(top_clone, top_data_gpu + (*top)[0]->offset(n),
	 sizeof(Dtype)*top_data_clone->count(),cudaMemcpyDeviceToDevice));
    caffe_matrix_transpose<Dtype>(top_clone, top_data_gpu + (*top)[0]->offset(n), N_, M_*GROUP_);
    // third, add bias
    if (biasterm_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
          N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
	//LOG(INFO)<<"finish forward.";
}

template <typename Dtype>
void UWConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * 1;
  int height_out = (HEIGHT_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  int width_out = (WIDTH_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  shared_ptr<Blob<Dtype> > top_data_clone;
  top_data_clone.reset(new Blob<Dtype>(MULTIPLY_BATCH, 1, N_, M_*GROUP_ ));
  Dtype* top_clone = top_data_clone->mutable_gpu_data();
  for (int n = 0; n < NUM_ / MULTIPLY_BATCH; ++n) {
    for (int i = 0; i < MULTIPLY_BATCH; ++i){
      // First, im2col
      im2col_uw_gpu(bottom_data+bottom[0]->offset(n * MULTIPLY_BATCH + i), 0, height_out, CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, PAD_, STRIDE_, col_data + col_buffer_.offset(i), i);
    }
    
    for (int i = 0; i < MULTIPLY_BATCH; ++i){
      for(int d=0;d<N_;d++){
        for (int g = 0; g < GROUP_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, 1, K_,
              (Dtype)1., weight + weight_offset * (d*GROUP_+g), col_data + col_buffer_.offset(i) + K_*(d*GROUP_+g),
              (Dtype)0., top_clone + top_data_clone->offset(i) + top_offset * (d * GROUP_+g), i);
        }
      }
    }
      
    for (int i = 0; i < MULTIPLY_BATCH; ++i){
      // Second, innerproduct with groups
      caffe_matrix_transpose<Dtype>(top_clone + top_data_clone->offset(i)
          , top_data + (*top)[0]->offset(n * MULTIPLY_BATCH + i), N_, M_*GROUP_, i);
    }
    
    // third, add bias
    if (biasterm_) {
      for (int i = 0; i < MULTIPLY_BATCH; ++i){
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
        N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        (Dtype)1., top_data + (*top)[0]->offset(n * MULTIPLY_BATCH + i), i);
      }
    }
    cudaDeviceSynchronize();
  }
}

template <typename Dtype>
Dtype UWConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  LOG(FATAL)<<"Backward_cpu not implemented.";
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_diff_gpu = top[0]->cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_data_gpu = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  Dtype* col_diff_gpu = col_buffer_.mutable_gpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (biasterm_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < NUM_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_, N_,
          1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
          bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  shared_ptr<Blob<Dtype> > bottom_data_clone;
  bottom_data_clone.reset(new Blob<Dtype>(1, 1, CHANNELS_ * KSIZE_ * KSIZE_, N_));
  Dtype* bottom_clone = bottom_data_clone->mutable_gpu_data();
  shared_ptr<Blob<Dtype> > bottom_diff_clone;
  bottom_diff_clone.reset(new Blob<Dtype>(1, 1, K_*GROUP_, N_));
  Dtype* diff_clone = bottom_diff_clone->mutable_gpu_data();
  shared_ptr<Blob<Dtype> > top_diff_clone;
  top_diff_clone.reset(new Blob<Dtype>(1, 1, M_*GROUP_, N_));
  Dtype* top_clone = top_diff_clone->mutable_gpu_data();
  for (int n = 0; n < NUM_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    //memset(bottom_diff +(*bottom)[0]->offset(n), 0, sizeof(Dtype)*(CHANNELS_*HEIGHT_*WIDTH_));
    im2col_cpu(bottom_data + (*bottom)[0]->offset(n), CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, PAD_, STRIDE_, col_data);
    caffe_matrix_transpose<Dtype>(col_data_gpu, bottom_clone, K_*GROUP_, N_);
    caffe_matrix_transpose<Dtype>(top_diff_gpu + top[0]->offset(n), top_clone, M_*GROUP_, N_);
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int d=0; d<N_; ++d){
    	for (int g = 0; g < GROUP_; ++g) {
      		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_*GROUP_, K_, 1,
        	(Dtype)1., top_diff_clone->mutable_cpu_data() + M_ * (d*GROUP_+g),
        	bottom_data_clone->mutable_cpu_data() + K_* (d*GROUP_+g), (Dtype)1.,
        	weight_diff + weight_offset * (d*GROUP_+g));
    	}
    }
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down) {
      caffe_matrix_transpose<Dtype>(col_diff_gpu, diff_clone, K_*GROUP_, N_);
      for (int d=0;d<N_; ++d){
      	for (int g = 0; g < GROUP_; ++g) {
        	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, M_,
          	(Dtype)1., top_diff_clone->mutable_cpu_data() + M_ * (d*GROUP_+g), 
		weight + weight_offset * (d*GROUP_+g),
          	(Dtype)0., bottom_diff_clone->mutable_cpu_data() + K_ * (d*GROUP_+g));
      	}
      }
      caffe_matrix_transpose<Dtype>(diff_clone, col_diff_gpu, N_, K_*GROUP_);
      // col2im back to the data
      col2im_cpu(col_diff, CHANNELS_, HEIGHT_,
          WIDTH_, KSIZE_, PAD_, STRIDE_, bottom_diff + (*bottom)[0]->offset(n));
    }
	LOG(INFO)<<"finish backward."<<n;
  }
  return Dtype(0.);
}

template <typename Dtype>
Dtype UWConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (biasterm_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0,
        sizeof(Dtype) * this->blobs_[1]->count()));
    for (int n = 0; n < NUM_ / MULTIPLY_BATCH; ++n) {
      for (int i = 0; i < MULTIPLY_BATCH; ++i){
        caffe_gpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_, N_,
            1., top_diff + top[0]->offset(n),
            reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
            1., bias_diff, i);
      }
    }
    cudaDeviceSynchronize();
  }

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  int height_out = (HEIGHT_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  int width_out = (WIDTH_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  CUDA_CHECK(cudaMemset(weight_diff, 0,
      sizeof(Dtype) * this->blobs_[0]->count()));
  shared_ptr<Blob<Dtype> > top_diff_clone;
  top_diff_clone.reset(new Blob<Dtype>(MULTIPLY_BATCH, 1, M_*GROUP_, N_));
  Dtype* top_clone = top_diff_clone->mutable_gpu_data();
  CUDA_CHECK(cudaMemset(bottom_diff, 0,
    sizeof(Dtype) * (NUM_ * CHANNELS_*HEIGHT_*WIDTH_)));
  for (int n = 0; n < NUM_ / MULTIPLY_BATCH; ++n) {
    for (int i = 0; i < MULTIPLY_BATCH; ++i){
      // since we saved memory in the forward pass by not storing all col data,
      // we will need to recompute them.
      caffe_matrix_transpose<Dtype>(top_diff + top[0]->offset(n * MULTIPLY_BATCH + i),
          top_clone + top_diff_clone->offset(i), M_*GROUP_, N_, i);
    }

    for (int i = 0; i < MULTIPLY_BATCH; ++i){
      im2col_uw_gpu(bottom_data + (*bottom)[0]->offset(n * MULTIPLY_BATCH + i),
        0, height_out, CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, PAD_, STRIDE_, col_data + col_buffer_.offset(i), i);
    }

   
      // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int i = 0; i < MULTIPLY_BATCH; ++i){
      for (int d=0; d<N_; ++d){
        for (int g = 0; g < GROUP_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, K_, 1,
          (Dtype)1., top_clone + top_diff_clone->offset(i) + M_ * (d*GROUP_+g),
          col_data + col_buffer_.offset(i) + K_* (d*GROUP_+g), (Dtype)1.,
          weight_diff + weight_offset * (d*GROUP_+g), i);
        }
      }
    }
    if (propagate_down) {
      for (int i = 0; i < MULTIPLY_BATCH; ++i){
      // gradient w.r.t. bottom data, if necessary
        for (int d=0;d<N_; ++d){
          for (int g = 0; g < GROUP_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, M_,
            (Dtype)1., top_clone + top_diff_clone->offset(i) + M_ * (d*GROUP_+g), 
            weight + weight_offset * (d*GROUP_+g),
            (Dtype)0., col_diff + col_buffer_.offset(i) + K_ * (d*GROUP_+g), i);
          }
        }
        // col2im back to the data
        col2im_uw_gpu(col_diff + col_buffer_.offset(i), 0, height_out, CHANNELS_, HEIGHT_,
          WIDTH_, KSIZE_, PAD_, STRIDE_, bottom_diff + (*bottom)[0]->offset(n * MULTIPLY_BATCH + i), i);
      }
    }
    cudaDeviceSynchronize();
  }
  return Dtype(0.);
}


template <typename Dtype>
void UWConvolutionLayer<Dtype>::RenormalizeFilter(){
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int MULTIPLY_BATCH = 8;
  if (this->layer_param_.has_max_rms()){
    Dtype rms = this->layer_param_.max_rms() * this->layer_param_.max_rms() *
      CHANNELS_ / GROUP_ * KSIZE_ * KSIZE_;
    for (int i = 0; i < NUM_ / MULTIPLY_BATCH; ++i){
      vector<Dtype> res(MULTIPLY_BATCH);
      for (int j = 0; j < MULTIPLY_BATCH; ++j){
        caffe_gpu_l2norm<Dtype>(CHANNELS_ / GROUP_ * KSIZE_ * KSIZE_,
          weight + this->blobs_[0]->offset(i), &res[j], j);
        Dtype ratio = rms / (res[j] * res[j]); 
        if (ratio < 1){
          caffe_gpu_scal(CHANNELS_ / GROUP_ * KSIZE_ * KSIZE_, ratio, 
            this->blobs_[0]->mutable_gpu_data(), j);
        }
      }
    }
    cudaDeviceSynchronize();
  }
}

INSTANTIATE_CLASS(UWConvolutionLayer);

}  // namespace caffe
