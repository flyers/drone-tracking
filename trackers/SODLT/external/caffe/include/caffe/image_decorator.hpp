#ifndef CAFFE_IMAGE_DECORATOR_HPP_
#define CAFFE_IMAGE_DECORATOR_HPP_

#include "caffe/common.hpp"
#include <opencv2/core/core.hpp>
#include "caffe/proto/caffe.pb.h"

using namespace std;
using namespace cv;
namespace caffe {

class ImageDecorator {
  public:
    virtual cv::Mat Decorate(const cv::Mat&) = 0;
  protected:
    shared_ptr<ImageDecorator> decorator;
};

class OriginalImageDecorator : public ImageDecorator{
  public:
    cv::Mat Decorate(const cv::Mat&);
};

class CropImageDecorator : public ImageDecorator{
  public:
    CropImageDecorator(const shared_ptr<ImageDecorator>& d, const string& side, int size = 0);
    cv::Mat Decorate(const cv::Mat&);
  protected:
    void GetUpperLeftCord(const string& spec, int width, int height, int cropsize,
      int& leftUpperW, int& leftUpperH);
    string side, pos;
    int cropsize;
    int leftUpperW, leftUpperH;
    int cropWidth, cropHeight;
};

class RandomCropImageDecorator : public ImageDecorator{
  public:
    RandomCropImageDecorator(const shared_ptr<ImageDecorator>& d, const int minSize,
      const int maxSize);
    cv::Mat Decorate(const cv::Mat&);
  protected:
    int minCropsize;
    int maxCropsize;
};


class ResizeImageDecorator : public ImageDecorator{
  public:
    ResizeImageDecorator(const shared_ptr<ImageDecorator>& d, int w, int h);
    cv::Mat Decorate(const cv::Mat&);
  protected:
    int width;
    int height;
};

class FlipImageDecorator : public ImageDecorator{
  public:
    FlipImageDecorator(const shared_ptr<ImageDecorator>& d, bool deterministic);
    cv::Mat Decorate(const cv::Mat&);
  protected:
    bool deterministic;
};

class LuminanceContrastVariationImageDecorator : public ImageDecorator{
  public:
    LuminanceContrastVariationImageDecorator(const shared_ptr<ImageDecorator>& d, 
        float l, float c);
    cv::Mat Decorate(const cv::Mat&);
  protected:
    float luminance_vary;
    float contrast_vary;
    void ConstructLookUp(float* mapping, const float& luminance, const float& contrast);
};

class BlurImageDecorator : public ImageDecorator{
  public:
    BlurImageDecorator(const shared_ptr<ImageDecorator>& d, 
        float r, float s);
    cv::Mat Decorate(const cv::Mat&);
  protected:
    float range;
    float sigma;
};

}  // namespace caffe

#endif
