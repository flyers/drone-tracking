// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <fcntl.h>

#include <string>
#include <vector>
#include <fstream>

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
void DataLayer<Dtype>::PrefetchForTest() {
  Datum datum;
  CHECK(prefetch_data_);
  CHECK(prefetch_label_);
  Dtype* top_data = prefetch_data_->mutable_cpu_data();
  Dtype* top_label = prefetch_label_->mutable_cpu_data();
  const Dtype scale = this->layer_param_.scale();
  int batchsize = this->layer_param_.batchsize();

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
    CHECK(this->iter_);
    CHECK(this->iter_->Valid());
    datum.ParseFromString(this->iter_->value().ToString());

    const string& data = datum.data();
    height = datum.height();
    width = datum.width();
    cv::Mat img = toMat(data, channels, height, width, height, width, 0, 0);
    int actualId = -1;
    int augmentationNum = augmentManager->GetSize(Caffe::phase());
    if (augmentationNum != 0){
      for (int j = 0; j < augmentationNum; ++j){
        cv::Mat img = augmentManager->Decorate(img, Caffe::phase(), j);
        CHECK(data.size()) << "Image augmentation only support uint8 data";
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
        top_label[actualId] = datum.label();
      }
    } else {
      // This corresponds to the non-image data.
      actualId = itemid;
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          Dtype datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[j]));
          top_data[itemid * size + j] = (datum_element - mean[j]) * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data[itemid * size + j] =
              (datum.float_data(j) - mean[j]) * scale;
        }
      }
      top_label[actualId] = datum.label();
    }
    
    // go to the next iter
    this->iter_->Next();
    if (!this->iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      this->iter_->SeekToFirst();
    }
  }
}


template <typename Dtype>
void DataLayer<Dtype>::PrefetchForTrainVal() {
  Datum datum;
  CHECK(prefetch_data_);
  Dtype* top_data = prefetch_data_->mutable_cpu_data();
  Dtype* top_label = prefetch_label_->mutable_cpu_data();
  const Dtype scale = this->layer_param_.scale();
  const int batchsize = this->layer_param_.batchsize();

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
    CHECK(iter_);
    CHECK(iter_->Valid());
    datum.ParseFromString(iter_->value().ToString());
    
    const string& data = datum.data();
    int height = datum.height();
    int width = datum.width();
    cv::Mat original, img;

    if (this->layer_param_.has_image_augmentation()) {
      CHECK(data.size()) << "Image augmentation only support uint8 data";
      CHECK_EQ(data_mean_.count(), outputSize * outputSize * channels);
      img = toMat(datum.data(), channels, height, width, height, width, 0, 0);
      img = augmentManager->Decorate(img, Caffe::phase());

      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < outputSize; ++h) {
          for (int w = 0; w < outputSize; ++w) {
            int index = ((itemid * channels + c) * outputSize + h) * outputSize + w;
            top_data[index] = (img.at<cv::Vec3b>(h,w)[c] - mean[(c * outputSize + h ) * outputSize + w]) * scale;
          }
        }
      }
    } else {
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          Dtype datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[j]));
          top_data[itemid * size + j] = (datum_element - mean[j]) * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data[itemid * size + j] =
              (datum.float_data(j) - mean[j]) * scale;
        }
      }
    }
    top_label[itemid] = datum.label();
    // go to the next iter
    iter_->Next();
    if (!iter_->Valid()) {
      // We have reached the end. Restart from the first.
      LOG(INFO) << "Restarting data prefetching from start.";
      iter_->SeekToFirst();
    }
  }
}

template <typename Dtype>
void DataLayer<Dtype>::CreatePrefetchThread() {
  CHECK(!StartInternalThread()) << "Pthread execution failed";
}

template <typename Dtype>
void DataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!WaitForInternalThreadToExit()) << "Pthread joining failed";
}

template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  if (Caffe::phase() == Caffe::TEST){
    // Only for test of image data.
    PrefetchForTest();
  } else {
    PrefetchForTrainVal();
  }
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
  // Finally, join the thread
  JoinPrefetchThread();
}

template <typename Dtype>
void DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_GT(this->layer_param_.output_size(), 0);
  this->GetNet()->SetBatchSize(this->layer_param_.batchsize());
  // Initialize the leveldb
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.source(), &db_temp);
	LOG(INFO)<<"open complete";
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.source() << std::endl << status.ToString();
  db_.reset(db_temp);
	leveldb::Iterator* itr;
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.rand_skip()) {
    unsigned int skip = rand() % this->layer_param_.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      iter_->Next();
      if (!iter_->Valid()) {
        iter_->SeekToFirst();
      }
    }
  }
  
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(iter_->value().ToString());

  ImageAugmentations augmentations;
  int augmentationSize = 1;
  // TODO here
  if((Caffe::phase() == Caffe::TEST) && (this->layer_param_.has_image_augmentation_list())){
    ReadProtoFromTextFile(this->layer_param_.image_augmentation_list(), &augmentations);
    augmentationSize = augmentations.image_augmentation_size();
  }
  outputSize = this->layer_param_.output_size();
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();
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
  (*top)[1]->Reshape(this->layer_param_.batchsize() * augmentationSize, 1, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.batchsize() * augmentationSize, 1, 1, 1));
  
  
  augmentManager.reset(new ImageAugmentationManager());

  // Set the mean values
  if(this->layer_param_.has_meanvalue()){
  	LOG(INFO) << "Loading mean value=" << this->layer_param_.meanvalue();
  	int count_ = datum.channels() * outputSize * outputSize;
  	data_mean_.Reshape(1, datum.channels(), outputSize, outputSize);
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
void DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
Dtype DataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype DataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
