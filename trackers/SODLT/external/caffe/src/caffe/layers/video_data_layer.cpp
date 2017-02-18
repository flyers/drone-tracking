// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <fcntl.h>

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/image_transform.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/common.hpp"
#include "caffe/augment_manager.hpp"
using namespace std;
using std::string;

namespace caffe {

template <typename Dtype>
void VideoDataLayer<Dtype>::PrefetchForTrainVal() {

  CHECK(prefetch_data_);
  Dtype* top_data = prefetch_data_->mutable_cpu_data();
  Dtype* top_label = prefetch_label_->mutable_cpu_data();
  const Dtype scale = this->layer_param_.scale();
  const int batchsize = this->layer_param_.batchsize();
  const int outputSize = this->layer_param_.output_size();
  const int shortEdge = this->layer_param_.short_edge();
  // datum scales
  const int channels = datum_channels_;
  const int size = datum_size_;
  const Dtype* mean = data_mean_.cpu_data();

  if (augmentManager->GetSize(Caffe::phase()) == 0 && 
    this->layer_param_.has_image_augmentation()){
    augmentManager->ParseConfig(this->layer_param_.image_augmentation(),
      Caffe::phase(), outputSize);
  }

  int isNeg;
  do {
    lines_id_ = rand() % videos_.size();
    isNeg = (videos_[lines_id_].second == 0);
  } while (isNeg && posCount / 2 < negCount);
  posCount += 1 - isNeg;
  negCount += isNeg;

  int actualFrameNum = videos_[lines_id_].first.size();
  vector<int> order;
  for (int i = 0; i < actualFrameNum; ++i){
    order.push_back(i);
  }
  random_shuffle(order.begin(), order.end());
  if (actualFrameNum > batchsize){
    actualFrameNum = min(actualFrameNum, batchsize);
    order.resize(actualFrameNum);
  }
  this->GetNet()->SetBatchSize(actualFrameNum);
  
  for (int itemid = 0; itemid < actualFrameNum; ++itemid) {
    // get a blob
    vector<std::string>& videos = videos_[lines_id_].first;
    cv::Mat img = ReadImageToMat(videos[order[itemid]], shortEdge);

    if (this->layer_param_.has_image_augmentation()) {
      CHECK_EQ(data_mean_.count(), outputSize * outputSize * channels);
      img = augmentManager->Decorate(img, Caffe::phase());
    }
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < outputSize; ++h) {
        for (int w = 0; w < outputSize; ++w) {
          int index = ((itemid * channels + c) * outputSize + h) * outputSize + w;
          top_data[index] = (img.at<cv::Vec3b>(h,w)[c] - mean[(c * outputSize + h ) * outputSize + w]) * scale;
        }
      }
    }
  }
  // This is for single label case.
  // top_label[0] = videos_[lines_id_].second;
  // This is for multilabel case. JUST A HACK HERE!
  for (int i = 0; i < 20; ++i){
    *(top_label + i) = 0;
  }
  if (videos_[lines_id_].second != 0){
    *(top_label + videos_[lines_id_].second - 1) = 1;
  }
  /*lines_id_++;
  if (lines_id_ >= videos_.size()) {
    // We have reached the end. Restart from the first.
    DLOG(INFO) << "Restarting data prefetching from start.";
    lines_id_ = 0;
    if (this->layer_param_.shuffle()) {
      random_shuffle(videos_.begin(), videos_.end());
    }
  }*/
}

template <typename Dtype>
void VideoDataLayer<Dtype>::CreatePrefetchThread() {
  CHECK(!StartInternalThread()) << "Pthread execution failed";
}

template <typename Dtype>
void VideoDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!WaitForInternalThreadToExit()) << "Pthread joining failed";
}

template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry() {
  PrefetchForTrainVal();
}

template <typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
  // Finally, join the thread
  JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_GT(this->layer_param_.output_size(), 0);
  this->GetNet()->SetBatchSize(this->layer_param_.batchsize());
  // Read the file with filenames and labels
  const string& source = this->layer_param_.source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string videoName;
  int numFrames, label;
  while (infile >> videoName >> label >> numFrames){
    vector<std::string> filenames(numFrames);
    for (int i = 0; i < numFrames; ++i){
      string temp;
      infile >> temp;
      filenames[i] = temp;
    }
    videos_.push_back(make_pair(filenames, label));
  }

  srand(time(NULL));
  if (this->layer_param_.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    random_shuffle(videos_.begin(), videos_.end());
  }

  LOG(INFO) << "A total of " << videos_.size() << " videos.";
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.rand_skip()) {
    unsigned int skip = rand() %
        this->layer_param_.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(videos_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  ImageAugmentations augmentations;
  int augmentationSize = 1;
  int posCount = 0;
  int negCount = 0;
  int outputSize = this->layer_param_.output_size();
  // datum size
  datum_channels_ = 3;
  datum_height_ = outputSize;
  datum_width_ = outputSize;
  datum_size_ = datum_channels_ * outputSize * outputSize;
  (*top)[0]->Reshape(
      this->layer_param_.batchsize(), datum_channels_, outputSize,
      outputSize);
  prefetch_data_.reset(new Blob<Dtype>(
      this->layer_param_.batchsize(), datum_channels_, outputSize,
      outputSize));
  
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // label, each batch only contains one single label.
  // HACK HERE FOR MULTILABEL CLS!!!
  (*top)[1]->Reshape(1, 20, 1, 1);
  prefetch_label_.reset(new Blob<Dtype>(1, 20, 1, 1));
  augmentManager.reset(new ImageAugmentationManager());

  // Set the mean values
  if(this->layer_param_.has_meanvalue()){
  	LOG(INFO) << "Loading mean value=" << this->layer_param_.meanvalue();
  	int count_ = datum_channels_* outputSize * outputSize;
  	data_mean_.Reshape(1, datum_channels_, outputSize, outputSize);
  	Dtype* data_vec = data_mean_.mutable_cpu_data();
  	for(int pos = 0; pos < count_; ++pos){
  		data_vec[pos] = this->layer_param_.meanvalue();
    }
  } else if (this->layer_param_.has_meanfile()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
    ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), (*top)[0]->channels());
    CHECK_EQ(data_mean_.height(), (*top)[0]->height());
    CHECK_EQ(data_mean_.width(), (*top)[0]->width());
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void VideoDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
      sizeof(Dtype) * prefetch_data_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());
	LOG(INFO)<<"data layer forward finished.";
  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
      prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
      cudaMemcpyHostToDevice));
  // Start a new prefetch thread
  CreatePrefetchThread();
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype VideoDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype VideoDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(VideoDataLayer);

}  // namespace caffe
