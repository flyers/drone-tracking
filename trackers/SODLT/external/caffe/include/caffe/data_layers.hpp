#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

//#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "leveldb/db.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/image_decorator.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/augment_manager.hpp"
namespace caffe {

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

// TODO: DataLayer, ImageDataLayer, and WindowDataLayer all have the
// same basic structure and a lot of duplicated code.

template <typename Dtype>
class DataLayer : public Layer<Dtype>, public InternalThread {
 public:
  explicit DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~DataLayer();
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

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  // The thread's function
  virtual void InternalThreadEntry();

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  shared_ptr<ImageAugmentationManager> augmentManager;

  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;

  int outputSize;
  int outputChannels;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
private:
  void PrefetchForTest();
  void PrefetchForTrainVal();
};

template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDF5DataLayer();
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
  virtual void LoadHDF5FileData(const char* filename);

  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

template <typename Dtype>
class HDF5OutputLayer : public Layer<Dtype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param);
  virtual ~HDF5OutputLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  inline std::string file_name() const { return file_name_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void SaveBlobs();

  std::string file_name_;
  hid_t file_id_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

template <typename Dtype>
class WindowDataLayer : public Layer<Dtype>, public InternalThread {
 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~WindowDataLayer();
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

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual void InternalThreadEntry();

  Blob<Dtype> prefetch_data_;
  Blob<Dtype> prefetch_label_;
  Blob<Dtype> prefetch_box_;
  Blob<Dtype> data_mean_;
  int sample_index_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
  shared_ptr<ImageAugmentationManager> augmentManager;
};

template <typename Dtype>
class ImageDataLayer : public Layer<Dtype>, public InternalThread {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~ImageDataLayer();
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

  virtual void ShuffleImages();

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual void InternalThreadEntry();

  vector<std::pair<std::string, vector<int> > > lines_;
  int lines_id_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  Caffe::Phase phase_;
  shared_ptr<ImageAugmentationManager> augmentManager;
private:
  void PrefetchForTest();
  void PrefetchForTrainVal();
};

template <typename Dtype>
class VideoDataLayer : public Layer<Dtype>, public InternalThread {
 public:
  explicit VideoDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~VideoDataLayer();
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

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual void InternalThreadEntry();

  vector<std::pair<vector<std::string>, int> > videos_;
  int lines_id_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  Caffe::Phase phase_;
  shared_ptr<ImageAugmentationManager> augmentManager;
  int posCount;
  int negCount;
private:
  void PrefetchForTrainVal();
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
