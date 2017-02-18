//
// Based on data_layer.cpp by Yangqing Jia.

#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/image_transform.hpp"
#include "caffe/augment_manager.hpp"
// caffe.proto > LayerParameter > WindowDataParameter
//   'source' field specifies the window_file
//   'outputSize' indicates the desired warped size

namespace caffe {

// Thread fetching the data
template <typename Dtype>
void WindowDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = prefetch_data_.mutable_cpu_data();
  Dtype* top_label = prefetch_label_.mutable_cpu_data();
  Dtype* top_box = prefetch_box_.mutable_cpu_data();
  const Dtype scale = this->layer_param_.scale();
  const int batch_size = this->layer_param_.batchsize();
  const int outputSize = this->layer_param_.output_size();
  const int context_pad = this->layer_param_.context_pad_pixels();
  const float fg_fraction =
      this->layer_param_.fg_fraction();
  const float context_pad_scale = this->layer_param_.context_pad_scale();
  const float output_scale = this->layer_param_.output_scale();
  const bool random_augment = this->layer_param_.random_augment();
  const bool output_multiple_label = this->layer_param_.output_multiple_label();
  const int num_label = this->layer_param_.num_label();
  const Dtype* mean = data_mean_.cpu_data();
  const int mean_off = (data_mean_.width() - outputSize) / 2;
  const int mean_width = data_mean_.width();
  const int mean_height = data_mean_.height();
  cv::Size cv_outputSize(outputSize, outputSize);
  const string& crop_mode = this->layer_param_.crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  caffe_set(prefetch_data_.count(), Dtype(0), top_data);
  caffe_set(prefetch_label_.count(), Dtype(0), top_label);
  const int num_fg = static_cast<int>(static_cast<float>(batch_size)
      * fg_fraction);
  const int num_samples[2] = { batch_size - num_fg, num_fg };

  int item_id = 0;
  // Parse the image augmentation config.
  if (augmentManager->GetSize(Caffe::phase()) == 0 && this->layer_param_.has_image_augmentation()){
    augmentManager->ParseConfig(this->layer_param_.image_augmentation(), Caffe::phase(), outputSize);
  }
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      const unsigned int rand_index = rand();
      vector<float>& window = (is_fg) ?
          fg_windows_[rand_index % fg_windows_.size()] :
          bg_windows_[rand_index % bg_windows_.size()];
      //LOG(INFO) << fg_windows_.size() << " " << bg_windows_.size() << " " << window[WindowDataLayer<Dtype>::IMAGE_INDEX];
      // load the image containing the window
      pair<std::string, vector<int> > image =
          image_database_[window[WindowDataLayer<Dtype>::IMAGE_INDEX]];

      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return;
      }
      const int channels = cv_img.channels();

      // crop window out of image and warp it
      int x1 = window[WindowDataLayer<Dtype>::X1];
      int y1 = window[WindowDataLayer<Dtype>::Y1];
      int x2 = window[WindowDataLayer<Dtype>::X2];
      int y2 = window[WindowDataLayer<Dtype>::Y2];

      int pad_w = 0;
      int pad_h = 0;
      // compute the expanded region
      Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
      Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
      Dtype center_x = static_cast<Dtype>(x1) + half_width;
      Dtype center_y = static_cast<Dtype>(y1) + half_height;
      if (use_square) {
        if (half_height > half_width) {
          half_width = half_height;
        } else {
          half_height = half_width;
        }
      }

      cv::Mat cv_cropped_img;
      if (!random_augment){

        // bool do_mirror = false;
        // if (mirror && rand() % 2) {
        //   do_mirror = true;
        // }
        if (context_pad > 0 || use_square) {
          // scale factor by which to expand the original region
          // such that after warping the expanded region to outputSize x outputSize
          // there's exactly context_pad amount of padding on each side
          Dtype context_scale = static_cast<Dtype>(outputSize) /
              static_cast<Dtype>(outputSize - 2*context_pad);

          
          x1 = static_cast<int>(round(center_x - half_width*context_scale));
          x2 = static_cast<int>(round(center_x + half_width*context_scale));
          y1 = static_cast<int>(round(center_y - half_height*context_scale));
          y2 = static_cast<int>(round(center_y + half_height*context_scale));

          // the expanded region may go outside of the image
          // so we compute the clipped (expanded) region and keep track of
          // the extent beyond the image
          int unclipped_height = y2-y1+1;
          int unclipped_width = x2-x1+1;
          int pad_x1 = std::max(0, -x1);
          int pad_y1 = std::max(0, -y1);
          int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
          int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
          // clip bounds
          x1 = x1 + pad_x1;
          x2 = x2 - pad_x2;
          y1 = y1 + pad_y1;
          y2 = y2 - pad_y2;
          CHECK_GT(x1, -1);
          CHECK_GT(y1, -1);
          CHECK_LT(x2, cv_img.cols);
          CHECK_LT(y2, cv_img.rows);

          int clipped_height = y2-y1+1;
          int clipped_width = x2-x1+1;

          // scale factors that would be used to warp the unclipped
          // expanded region
          Dtype scale_x =
              static_cast<Dtype>(outputSize)/static_cast<Dtype>(unclipped_width);
          Dtype scale_y =
              static_cast<Dtype>(outputSize)/static_cast<Dtype>(unclipped_height);

          // size to warp the clipped expanded region to
          cv_outputSize.width =
              static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
          cv_outputSize.height =
              static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
          pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
          pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
          pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
          pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

          pad_h = pad_y1;
          pad_w = pad_x1;
          // // if we're mirroring, we mirror the padding too (to be pedantic)
          // if (do_mirror) {
          //   pad_w = pad_x2;
          // } else {
          //   pad_w = pad_x1;
          // }

          // ensure that the warped, clipped region plus the padding fits in the
          // outputSize x outputSize image (it might not due to rounding)
          if (pad_h + cv_outputSize.height > outputSize) {
            cv_outputSize.height = outputSize - pad_h;
          }
          if (pad_w + cv_outputSize.width > outputSize) {
            cv_outputSize.width = outputSize - pad_w;
          }
        }

        cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
        cv_cropped_img = cv_img(roi);
        
        if (augmentManager->GetSize(Caffe::phase()) != 0){ 
          cv_cropped_img = augmentManager->Decorate(cv_cropped_img, Caffe::phase());
        } 
        resizeImage(cv_cropped_img, cv_cropped_img, cv_outputSize.width, cv_outputSize.height);
        
        // copy the warped window into top_data
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < cv_cropped_img.rows; ++h) {
            for (int w = 0; w < cv_cropped_img.cols; ++w) {
              Dtype pixel =
                  static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

              top_data[((item_id * channels + c) * outputSize + h + pad_h)
                       * outputSize + w + pad_w]
                  = (pixel
                      - mean[(c * mean_height + h + mean_off + pad_h)
                             * mean_width + w + mean_off + pad_w])
                    * scale;
            }
          }
        }

        // // horizontal flip at random
        // if (do_mirror) {
        //   cv::flip(cv_cropped_img, cv_cropped_img, 1);
        // }
      } else {        
        if (window[WindowDataLayer<Dtype>::LABEL] != 0){
          Dtype objX1 = window[WindowDataLayer<Dtype>::X1];
          Dtype objY1 = window[WindowDataLayer<Dtype>::Y1];
          Dtype objX2 = window[WindowDataLayer<Dtype>::X2];
          Dtype objY2 = window[WindowDataLayer<Dtype>::Y2];
          //LOG(INFO) << objX1 << " " << objY1 << " " << objX2 << " " << objY2;
          float context_scale = (double)rand() / RAND_MAX * context_pad_scale;
          context_scale += 1.2;
          
          x1 = static_cast<int>(round(center_x - half_width*context_scale));
          x2 = static_cast<int>(round(center_x + half_width*context_scale));  
          y1 = static_cast<int>(round(center_y - half_height*context_scale));
          y2 = static_cast<int>(round(center_y + half_height*context_scale));

          Dtype translationWidth, translationHeight;
          translationWidth = rand() % int(round((context_scale - 1) * half_width) + 1);
          translationHeight = rand() % int(round((context_scale - 1) * half_height) + 1);
          translationWidth = (rand() % 2 ? 1 : -1) * translationWidth;
          translationHeight = (rand() % 2 ? 1 : -1) * translationHeight;

          x1 += translationWidth;
          x2 += translationWidth;
          y1 += translationHeight;
          y2 += translationHeight;

          int pad_x1 = std::max(0, -x1);
          int pad_y1 = std::max(0, -y1);
          int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
          int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
          if (!(pad_x1 == 0 && pad_y1 == 0 && pad_x2 == 0 && pad_y2 == 0)){
            //DLOG(INFO) << "Before: " << cv_img.cols << " " << cv_img.rows;
            cv::copyMakeBorder(cv_img, cv_img, pad_y1, pad_y2, pad_x1, pad_x2, cv::BORDER_CONSTANT);
            //DLOG(INFO) << "After: " << cv_img.cols << " " << cv_img.rows;
          }

          objX1 -= x1;
          objY1 -= y1;
          objX2 -= x1;
          objY2 -= y1;
          x1 += pad_x1;
          x2 += pad_x1;
          y1 += pad_y1;
          y2 += pad_y1;
          x2 = min(x2, cv_img.cols);
          y2 = min(y2, cv_img.rows);
          cv::Rect roi(x1, y1, x2-x1, y2-y1);
          //LOG(INFO) << x1 << " " << y1 << " " << x2 << " " << y2 << " " << cv_img.cols << " " << cv_img.rows;
          cv_cropped_img = cv_img(roi);

          top_box[item_id * 4] = ceil(objX1 / cv_cropped_img.cols * outputSize / output_scale);
          top_box[item_id * 4 + 1] = floor(objY1 / cv_cropped_img.rows * outputSize / output_scale);
          top_box[item_id * 4 + 2] = ceil(objX2 / cv_cropped_img.cols * outputSize / output_scale);
          top_box[item_id * 4 + 3] = floor(objY2 / cv_cropped_img.rows * outputSize / output_scale);
        } else {
          cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
          cv_cropped_img = cv_img(roi);
          resizeImage(cv_cropped_img, cv_cropped_img, outputSize, outputSize);

          top_box[item_id * 4] = 0;
          top_box[item_id * 4 + 1] = 0;
          top_box[item_id * 4 + 2] = -1;
          top_box[item_id * 4 + 3] = -1;
        }
        resizeImage(cv_cropped_img, cv_cropped_img, outputSize, outputSize);
        if (augmentManager->GetSize(Caffe::phase()) != 0){
          cv_cropped_img = augmentManager->Decorate(cv_cropped_img, Caffe::phase());
        }
        // copy the warped window into top_data
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < cv_cropped_img.rows; ++h) {
            for (int w = 0; w < cv_cropped_img.cols; ++w) {
              Dtype pixel =
                  static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

              top_data[((item_id * channels + c) * outputSize + h) * outputSize + w]
                  = (pixel - mean[(c * mean_height + h + mean_off)
                             * mean_width + w + mean_off])
                    * scale;
            }
          }
        }
      }

      // get window label
      if (!output_multiple_label){
          top_label[item_id] = window[WindowDataLayer<Dtype>::LABEL];
      } else {
          if (window[WindowDataLayer<Dtype>::LABEL] != 0){
            top_label[item_id * num_label + static_cast<int>(
              window[WindowDataLayer<Dtype>::LABEL]) - 1] = 1;
          }
      }

      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << sample_index_++;
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << '\n'
          << top_box[item_id * 4] << '\n'
          << top_box[item_id * 4 + 1] << '\n'
          << top_box[item_id * 4 + 2] << '\n'
          << top_box[item_id * 4 + 3] << '\n'
          << top_label[item_id] << '\n'
          << is_fg << std::endl;
      inf.close();

      cv::imwrite("dump/" + file_id + ".jpg", cv_cropped_img);
      #endif

      item_id++;
    }
  }
}

template <typename Dtype>
WindowDataLayer<Dtype>::~WindowDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void WindowDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // SetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  LOG(INFO) << "Window data layer:" << '\n'
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.fg_threshold() << '\n'
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.bg_threshold() << '\n'
      << "  foreground sampling fraction: "
      << this->layer_param_.fg_fraction();

  this->GetNet()->SetBatchSize(this->layer_param_.batchsize());
  sample_index_ = 0;
  std::ifstream infile(this->layer_param_.source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.source() << '\n';

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    image_database_.push_back(std::make_pair(image_path, image_size));

    // read each box
    int num_windows;
    infile >> num_windows;
    const float fg_threshold =
        this->layer_param_.fg_threshold();
    const float bg_threshold =
        this->layer_param_.bg_threshold();
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;

      vector<float> window(WindowDataLayer::NUM);
      window[WindowDataLayer::IMAGE_INDEX] = image_index;
      window[WindowDataLayer::LABEL] = label;
      window[WindowDataLayer::OVERLAP] = overlap;
      window[WindowDataLayer::X1] = x1;
      window[WindowDataLayer::Y1] = y1;
      window[WindowDataLayer::X2] = x2;
      window[WindowDataLayer::Y2] = y2;

      // add window to foreground list or background list
      if (overlap >= fg_threshold) {
        int label = window[WindowDataLayer::LABEL];
        CHECK_GT(label, 0);
        fg_windows_.push_back(window);
        label_hist.insert(std::make_pair(label, 0));
        label_hist[label]++;
      } else if (overlap < bg_threshold) {
        if ((y2 - y1) * (x2 - x1) > 400){
          // background window, force label and overlap to 0
          window[WindowDataLayer::LABEL] = 0;
          window[WindowDataLayer::OVERLAP] = 0;
          bg_windows_.push_back(window);
          label_hist[0]++;
        }
      }
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  LOG(INFO) << "Amount of context padding (pixels): "
      << this->layer_param_.context_pad_pixels();

  LOG(INFO) << "Amount of context padding (scale): "
    << this->layer_param_.context_pad_scale();

  LOG(INFO) << "Crop mode: "
      << this->layer_param_.crop_mode();

  // image
  int outputSize = this->layer_param_.output_size();
  CHECK_GT(outputSize, 0);
  const int batch_size = this->layer_param_.batchsize();
  (*top)[0]->Reshape(batch_size, channels, outputSize, outputSize);
  prefetch_data_.Reshape(batch_size, channels, outputSize, outputSize);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  const bool output_multiple_label = this->layer_param_.output_multiple_label();
  const int num_label = this->layer_param_.num_label();
  if (output_multiple_label){
    (*top)[1]->Reshape(batch_size, num_label, 1, 1);
    prefetch_label_.Reshape(batch_size, num_label, 1, 1);  
  } else {
    (*top)[1]->Reshape(batch_size, 1, 1, 1);
    prefetch_label_.Reshape(batch_size, 1, 1, 1);
  }

  if (top->size() > 2){
    (*top)[2]->Reshape(batch_size, 4, 1, 1);
  }
  prefetch_box_.Reshape(batch_size, 4, 1, 1);

  // check if we want to have mean
  if (this->layer_param_.has_meanfile()) {
    const string& mean_file =
        this->layer_param_.meanfile();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFile(mean_file, &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.width(), data_mean_.height());
    CHECK_EQ(data_mean_.channels(), channels);
  } else if (this->layer_param_.has_meanvalue()) {
    LOG(INFO) << "Loading mean value = " << this->layer_param_.meanvalue();
    int count_ = channels * outputSize * outputSize;
    data_mean_.Reshape(1, channels, outputSize, outputSize);
    Dtype* data_vec = data_mean_.mutable_cpu_data();
    for(int pos = 0; pos < count_; ++pos){
      data_vec[pos] = this->layer_param_.meanvalue();
    }
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, channels, outputSize, outputSize);
  }

  augmentManager.reset(new ImageAugmentationManager());
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_.mutable_cpu_data();
  prefetch_label_.mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void WindowDataLayer<Dtype>::CreatePrefetchThread() {
  // Create the thread.
  CHECK(!StartInternalThread()) << "Pthread execution failed.";
}

template <typename Dtype>
void WindowDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!WaitForInternalThreadToExit()) << "Pthread joining failed.";
}

template <typename Dtype>
void WindowDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
             (*top)[1]->mutable_cpu_data());
  if (top->size() > 2){
    caffe_copy(prefetch_box_.count(), prefetch_box_.cpu_data(),
             (*top)[2]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype WindowDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(WindowDataLayer);

}  // namespace caffe
