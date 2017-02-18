// Copyright 2013 Yangqing Jia


#include <mkl.h>
#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Mask Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Mask Layer takes one blob as output.";
  (*top)[0]->Reshape(bottom[0]->num() * bottom[0]->channels(), bottom[1]->channels(),
    bottom[1]->height(), bottom[1]->width());
  targets_.reset(new Blob<Dtype>(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(),
    bottom[0]->width()));
  // (*top)[1]->Reshape(bottom[0]->num() * bottom[0]->channels(), bottom[2]->channels(),
  //   bottom[2]->height(), bottom[2]->width());
  output_scale_ = this->layer_param_.output_scale();
};

template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
Dtype MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
  return Dtype(0.0);
}

template <typename Dtype>
__global__ void MaskLayerForward(const int nthreads, const int num, const int classNum,
  const int height, const int width, const int channelOut, int outputScale, const Dtype* mask,
  const Dtype* data, Dtype* out){
  int index = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x;
  if (index < nthreads){
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % classNum;
    int n = index / width / height / classNum;
    int heightOut = height * outputScale;
    int widthOut = width * outputScale;
    int outN = n * classNum + c;
    int outH = h * outputScale;
    int outW = w * outputScale;
    for (int i = 0; i < channelOut; ++i){
      for (int j = outH; j < outH + outputScale; ++j){
        for (int k = outW; k < outW + outputScale; ++k){
          int id = outN * (channelOut * heightOut * widthOut) + i * heightOut * widthOut
            + j * widthOut + k;
          int imageId = n * (channelOut * heightOut * widthOut) + i * heightOut * widthOut
            + j * widthOut + k;
          out[id] = mask[index] * (data[imageId] + 120) - 120;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void MaskLayerBackward(const int nthreads, const int num, const int classNum,
  const int height, const int width, const int channelOut, int outputScale, const Dtype* top_diff,
  const Dtype* data, const Dtype* mask, Dtype* bottom_diff, Dtype* targets){
  int index = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x;
  if (index < nthreads){
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % classNum;
    int n = index / width / height / classNum;
    int heightOut = height * outputScale;
    int widthOut = width * outputScale;
    int outN = n * classNum + c;
    int outH = h * outputScale;
    int outW = w * outputScale;
    Dtype grad = 0;
    for (int i = 0; i < channelOut; ++i){
      for (int j = outH; j < outH + outputScale; ++j){
        for (int k = outW; k < outW + outputScale; ++k){
          int id = outN * (channelOut * heightOut * widthOut) + i * heightOut * widthOut
            + j * widthOut + k;
          int imageId = n * (channelOut * heightOut * widthOut) + i * heightOut * widthOut
            + j * widthOut + k;
          grad += abs(top_diff[id]);
        }
      }
    }
    grad *= 50;
    grad = min(grad, 1.0);
    targets[index] = grad;
    bottom_diff[index] = (mask[index] - grad) / 1e4 * (grad != 0 ? 10 : 0.1);
  }
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  MaskLayerForward<Dtype><<<CAFFE_GET_BLOCKS2D(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
    bottom[1]->channels(), output_scale_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
    (*top)[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  this->GetNet()->SetBatchSize((*top)[0]->num());

  
  // // We need to duplicate the labels.
  // for (int i = 0; i < bottom[2]->num(); ++i){
  //   for (int j = 0; j < bottom[2]->channels(); ++j){
  //     caffe_copy(bottom[2]->channels(), bottom[2]->gpu_data() + bottom[2]->offset(i, 0, 0, 0),
  //       (*top)[1]->mutable_gpu_data() + (*top)[1]->offset(i * bottom[2]->channels() + j, 0, 0, 0));
  //   }
  // }
  // for (int k = 0; k < 2; ++k){
  //   for (int i = 0; i < 20; ++i){
  //     LOG(INFO) << (bottom)[0]->data_at(k, i, 10, 10);
  //   }
  // }
  for (int i = 0; i < 20; ++i){
    LOG(INFO) << (bottom)[0]->diff_at(0, i, 4, 4);
  }

}

template <typename Dtype>
Dtype MaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  int count = (*bottom)[0]->count();
  MaskLayerBackward<Dtype><<<CAFFE_GET_BLOCKS2D(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, (*bottom)[0]->num(), (*bottom)[0]->channels(), (*bottom)[0]->height(), (*bottom)[0]->width(),
    (*bottom)[1]->channels(), output_scale_, top[0]->gpu_diff(), (*bottom)[1]->gpu_data(),
    (*bottom)[0]->gpu_data(), (*bottom)[0]->mutable_gpu_diff(), targets_->mutable_gpu_diff());
  // for (int i = 0; i < 20; ++i){
  //   LOG(INFO) << (top)[0]->diff_at(i, 0, 100, 100);
  // }
  static int sample_index_ = 0;
  string file_id;
  std::stringstream ss;
  ss << sample_index_++;
  ss >> file_id;

  cv::Mat img = Mat::zeros(224, 224, CV_8UC3);
  for(int c = 0;c < 3; ++c){
    for(int h = 0; h < 224; ++h){
      for(int w = 0; w < 224;++w){
        int x = abs((top)[0]->diff_at(0, c, h, w) * 1e8);
        x = x < 0 ? 0 : x;
        img.at<cv::Vec3b>(h, w)[c] = (x > 255 ? 255 : x);
      }
    }
  }
  cv::imwrite("dump/" + file_id + "_top.jpg", img);


  img = Mat::zeros(224, 224, CV_8UC3);
  for(int c = 0;c < 3; ++c){
    for(int h = 0; h < 224; ++h){
      for(int w = 0; w < 224;++w){
        int x = (top)[0]->data_at(0, c, h, w) + 120;
        x = x < 0 ? 0 : x;
        img.at<cv::Vec3b>(h, w)[c] = (x > 255 ? 255 : x);
      }
    }
  }
  cv::imwrite("dump/" + file_id + "_raw.jpg", img);
  
  img = Mat::zeros(7, 7, CV_8UC3);
  for(int c = 0;c < 3; ++c){
    for(int h = 0; h < 7; ++h){
      for(int w = 0; w < 7;++w){
        float x = targets_->diff_at(0, 0, h, w);
        x = x < 0 ? 0 : x;
        img.at<cv::Vec3b>(h, w)[c] = (x * 255 > 255 ? 255 : x * 255);
      }
    }
  }
  cv::imwrite("dump/" + file_id + ".jpg", img);
  CUDA_POST_KERNEL_CHECK;
  this->GetNet()->SetBatchSize((*bottom)[0]->num());
  return Dtype(0.0);
}

INSTANTIATE_CLASS(MaskLayer);

}  // namespace caffe
