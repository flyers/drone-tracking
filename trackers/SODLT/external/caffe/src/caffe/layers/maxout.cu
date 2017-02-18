#include <algorithm>
#include <cfloat>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
__global__ void MaxoutForward(const int nthreads,
    const Dtype* temp_data, const int k, Dtype* top_data, Dtype* top_index) {
  int index = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
  if (index < nthreads) {
    Dtype out = temp_data[index * k];
    float out_index = index * k;
    for (int i = index * k + 1; i < index * k + k; ++i){
      if (temp_data[i] > out){
        out = temp_data[i];
        out_index = i;
      }
    }
    top_data[index] = out;
    top_index[index] = out_index;
  }  // (if index < nthreads)
}

template <typename Dtype>
__global__ void SetToPosition(const int nthreads, const int dim,
    const Dtype* source, const int k, Dtype* dest, const Dtype* top_index) {
  int index = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
  if (index < nthreads) {
    for (int i = 0; i < dim; ++i){
      dest[i * nthreads * k + (int)top_index[index]] = source[i * nthreads + index];
    }
    //memcpy((dest + (int)top_index[index] * dim), (source + index * dim),
    // sizeof(Dtype) * dim);
  }  // (if index < nthreads)
}

template <typename Dtype>
void MaxoutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Maxout Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Maxout Layer takes a single blob as output.";
  CHECK_GT(this->layer_param_.k_per_group(), 0) << "k_per_group must be positive.";
  const int num_output = this->layer_param_.num_output();
  biasterm_ = this->layer_param_.biasterm();
  // Figure out the dimensions
  M_ = bottom[0]->num(); // Batchsize
  K_ = bottom[0]->count() / bottom[0]->num(); // Output dim of previous layer
  N_ = num_output;
  K_PER_GROUP = this->layer_param_.k_per_group();

  CHECK_EQ(N_ % K_PER_GROUP, 0) <<
    "The number of output in Maxout layer must be divisible by k_per_group.";

  (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (biasterm_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_ * K_PER_GROUP, K_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(
        GetFiller<Dtype>(this->layer_param_.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (biasterm_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_ * K_PER_GROUP));
      shared_ptr<Filler<Dtype> > bias_filler(
          GetFiller<Dtype>(this->layer_param_.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  // Setting up the bias multiplier
  if (biasterm_) {
    bias_multiplier_.reset(new SyncedMemory(M_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < M_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
  // Setup the buffer for temp results;
  buffer_out.reset(new SyncedMemory(M_ * K_PER_GROUP * N_ * sizeof(Dtype)));
  out_index.reset(new SyncedMemory(M_ * N_ * sizeof(Dtype)));
};

template <typename Dtype>
void MaxoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  NOT_IMPLEMENTED;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (biasterm_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
Dtype MaxoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  // Gradient with respect to weight
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
      top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
  if (biasterm_) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down) {
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_cpu_diff());
  }
  return Dtype(0);
}

template <typename Dtype>
void MaxoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_ * K_PER_GROUP, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., (Dtype*)(buffer_out->mutable_gpu_data()));
  if (biasterm_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_ * K_PER_GROUP, 1, (Dtype)1.,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        this->blobs_[1]->gpu_data(), (Dtype)1.,
        (Dtype*)(buffer_out->mutable_gpu_data()));
  }
  MaxoutForward<Dtype><<<CAFFE_GET_BLOCKS2D(M_ * N_), CAFFE_CUDA_NUM_THREADS>>>(
    M_ * N_, (Dtype*)(buffer_out->mutable_gpu_data()), 
    K_PER_GROUP, (*top)[0]->mutable_gpu_data(),
    (Dtype*)(out_index->mutable_gpu_data()));
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
Dtype MaxoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* buffer = (Dtype*)(buffer_out->mutable_gpu_data());
  cudaMemset(buffer, 0, M_ * K_PER_GROUP * N_ * sizeof(Dtype));
  SetToPosition<Dtype><<<CAFFE_GET_BLOCKS2D(N_), CAFFE_CUDA_NUM_THREADS>>>(
    N_, M_, top_diff, K_PER_GROUP, buffer,
    (Dtype*)(out_index->mutable_gpu_data()));
  CUDA_POST_KERNEL_CHECK;
  // Gradient with respect to weight
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_ * K_PER_GROUP, K_, M_, (Dtype)1.,
      buffer, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  if (biasterm_) {
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_ * K_PER_GROUP, (Dtype)1., buffer,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        (Dtype)0., this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down) {
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_ * K_PER_GROUP, (Dtype)1.,
        buffer, this->blobs_[0]->gpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_gpu_diff());
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(MaxoutLayer);

} // namespace caffe