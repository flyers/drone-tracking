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
void ImageDataLayer<Dtype>::PrefetchForTest() {
  CHECK(prefetch_data_);
  CHECK(prefetch_label_);
  Dtype* top_data = prefetch_data_->mutable_cpu_data();
  Dtype* top_label = prefetch_label_->mutable_cpu_data();
  const Dtype scale = this->layer_param_.scale();
  int batchsize = this->layer_param_.batchsize();
  int shortEdge = this->layer_param_.short_edge();
  const int outputSize = this->layer_param_.output_size();
  const int lines_size = lines_.size();
  // datum scales
  const int channels = datum_channels_;
  const int size = datum_size_;
  const Dtype* mean = data_mean_.cpu_data();

  ImageAugmentations augmentations;
  int imagesPerBatch = batchsize;
  if (augmentManager->GetSize(Caffe::phase()) == 0){
    if (this->layer_param_.has_image_augmentation_list()){
      ReadProtoFromTextFile(this->layer_param_.image_augmentation_list(), &augmentations);
      batchsize *= augmentations.image_augmentation_size();
      for (int i = 0; i < augmentations.image_augmentation_size(); ++i){
        augmentManager->ParseConfig(augmentations.image_augmentation(i), Caffe::phase(), outputSize);
      }
    } else {
      LOG(INFO) << "No augmentation configurations, treat as non-image data.";
    }
  }

  int height,width;
  
  for (int itemid = 0; itemid < imagesPerBatch; itemid++){
    // get a blob
    cv::Mat img = ReadImageToMat(lines_[lines_id_].first, shortEdge);
    int actualId = -1;
    int augmentationNum = augmentManager->GetSize(Caffe::phase());
    if (augmentationNum != 0){
      for (int j = 0; j < augmentationNum; ++j){
        img = augmentManager->Decorate(img, Caffe::phase(), j);
        actualId = itemid * augmentationNum + j;
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < outputSize; ++h) {
            for (int w = 0; w < outputSize; ++w) {
              top_data[((actualId * channels + c) * outputSize + h) * outputSize + w]
                  = (static_cast<Dtype>((uint8_t)img.at<cv::Vec3b>(h,w)[c])
                    - mean[(c * outputSize + h ) * outputSize + w]) * scale;
            }
          }
        }
        for (int k = 0; k < lines_[lines_id_].second.size(); ++k){
          *(top_label + prefetch_label_->offset(actualId, k, 0, 0)) = lines_[lines_id_].second[k];
        }
      }
    }
    
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      LOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.shuffle()) {
        ShuffleImages();
      }
    }
  }
}


template <typename Dtype>
void ImageDataLayer<Dtype>::PrefetchForTrainVal() {

  CHECK(prefetch_data_);
  Dtype* top_data = prefetch_data_->mutable_cpu_data();
  Dtype* top_label = prefetch_label_->mutable_cpu_data();
  const Dtype scale = this->layer_param_.scale();
  const int batchsize = this->layer_param_.batchsize();
  const int outputSize = this->layer_param_.output_size();
  const int lines_size = lines_.size();
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
  for (int itemid = 0; itemid < batchsize; ++itemid) {
    // get a blob
    cv::Mat original = ReadImageToMat(lines_[lines_id_].first, shortEdge);
    cv::Mat img;

    if (this->layer_param_.has_image_augmentation()) {
      CHECK_EQ(data_mean_.count(), outputSize * outputSize * channels);
      img = augmentManager->Decorate(original, Caffe::phase());
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < outputSize; ++h) {
          for (int w = 0; w < outputSize; ++w) {
            int index = ((itemid * channels + c) * outputSize + h) * outputSize + w;
            top_data[index] = (img.at<cv::Vec3b>(h,w)[c] - mean[(c * outputSize + h ) * outputSize + w]) * scale;
          }
        }
      }
    }
    for (int k = 0; k < lines_[lines_id_].second.size(); ++k){
      *(top_label + prefetch_label_->offset(itemid, k, 0, 0)) = lines_[lines_id_].second[k];
    }
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.shuffle()) {
        ShuffleImages();
      }
    }
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::CreatePrefetchThread() {
  CHECK(!StartInternalThread()) << "Pthread execution failed";
}

template <typename Dtype>
void ImageDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!WaitForInternalThreadToExit()) << "Pthread joining failed";
}

template <typename Dtype>
void ImageDataLayer<Dtype>::InternalThreadEntry() {
  if (Caffe::phase() == Caffe::TEST){
    // Only for test of image data.
    PrefetchForTest();
  } else {
    PrefetchForTrainVal();
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  random_shuffle(lines_.begin(), lines_.end());
}

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  // Finally, join the thread
  JoinPrefetchThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_GT(this->layer_param_.output_size(), 0);
  this->GetNet()->SetBatchSize(this->layer_param_.batchsize());
  // Read the file with filenames and labels
  const string& source = this->layer_param_.source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int labelNum;
  infile >> labelNum;
  LOG(INFO) << "The dataset contains " << labelNum << " labels.";
  while (infile >> filename){
    vector<int> labels;
    int label;
    for (int i = 0; i < labelNum; ++i){
      infile >> label;
      labels.push_back(label);
    }
    lines_.push_back(std::make_pair(filename, labels));
  }
  srand(time(NULL));
  if (this->layer_param_.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    ShuffleImages();
  }

  LOG(INFO) << "A total of " << lines_.size() << " images.";
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.rand_skip()) {
    unsigned int skip = rand() %
        this->layer_param_.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  ImageAugmentations augmentations;
  int augmentationSize = 1;
  // TODO here
  if((Caffe::phase() == Caffe::TEST) && (this->layer_param_.has_image_augmentation_list())){
    ReadProtoFromTextFile(this->layer_param_.image_augmentation_list(), &augmentations);
    augmentationSize = augmentations.image_augmentation_size();
  }
  int outputSize = this->layer_param_.output_size();
  // datum size
  datum_channels_ = 3;
  datum_height_ = outputSize;
  datum_width_ = outputSize;
  datum_size_ = datum_channels_ * outputSize * outputSize;
  (*top)[0]->Reshape(
      this->layer_param_.batchsize() * augmentationSize, datum_channels_, outputSize,
      outputSize);
  prefetch_data_.reset(new Blob<Dtype>(
      this->layer_param_.batchsize() * augmentationSize, datum_channels_, outputSize,
      outputSize));
  
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(this->layer_param_.batchsize() * augmentationSize, labelNum, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.batchsize() * augmentationSize, labelNum, 1, 1));
  
  
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
void ImageDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void ImageDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
Dtype ImageDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype ImageDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImageDataLayer);

}  // namespace caffe
