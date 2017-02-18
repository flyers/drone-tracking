#include "caffe/image_decorator.hpp"
#include "caffe/util/image_transform.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
  cv::Mat OriginalImageDecorator::Decorate(const cv::Mat& img){
    return img;
  }

  FlipImageDecorator::FlipImageDecorator(const shared_ptr<ImageDecorator>& d, bool deterministic = false)
     : deterministic(deterministic) {
      decorator = d;
    }

  cv::Mat FlipImageDecorator::Decorate(const cv::Mat& ori){
  	cv::Mat img = decorator->Decorate(ori);
  	if (deterministic || rand() % 2){
  		flipImage(img, img); 
  	}
  	return img;
  }

  ResizeImageDecorator::ResizeImageDecorator(const shared_ptr<ImageDecorator>& d, 
    int w, int h) : width(w), height(h){
      decorator = d;
    }
  cv::Mat ResizeImageDecorator::Decorate(const cv::Mat& ori){
    //LOG(INFO) << ori.cols << " " << ori.rows << " " << width << " " << height;
  	cv::Mat img = decorator->Decorate(ori);
  	resizeImage(img, img, width, height);
  	return img;
  }

	CropImageDecorator::CropImageDecorator(const shared_ptr<ImageDecorator>& d, const string& side,
      int size) : side(side), cropsize(size){
      decorator = d;
    }

  cv::Mat CropImageDecorator::Decorate(const cv::Mat& ori){
  	cv::Mat img = decorator->Decorate(ori);
  	if (side == "full"){
  		return img;
  	}
  	int leftUpperW, leftUpperH;
  	int oriWidth = img.size().width;
  	int oriHeight = img.size().height;
  	int size = cropsize;
  	if (size == 0){
  		size = min(oriWidth, oriHeight);
  	}
  	GetUpperLeftCord(side, oriWidth, oriHeight, size, leftUpperW, leftUpperH);
    //LOG(INFO) << ori.cols << " " << ori.rows << " " << size;

  	cropImage(img, img, leftUpperW, leftUpperH, size, size);
  	return img;
  }

  void CropImageDecorator::GetUpperLeftCord(const string& spec, int width, int height, int cropsize,
      int& leftUpperW, int& leftUpperH){
	    if(spec == "left" || spec == "leftup"){
	      leftUpperH = 0;
	      leftUpperW = 0;
	    }
	    else if(spec == "middle"){
	      leftUpperH = (height - cropsize) / 2;
	      leftUpperW = (width - cropsize) / 2;
	    }
	    else if(spec == "right" || spec == "rightbot"){
	      leftUpperH = (height - cropsize);
	      leftUpperW = (width - cropsize);
	    } else if (spec == "rightup"){
	      leftUpperH = 0;
	      leftUpperW = (width - cropsize);
	    } else if (spec == "leftbot"){
	      leftUpperH = height - cropsize;
	      leftUpperW = 0;
	    }
	}

  RandomCropImageDecorator::RandomCropImageDecorator(const shared_ptr<ImageDecorator>& d, 
      const int minSize, const int maxSize) : minCropsize(minSize),
      maxCropsize(maxSize) {
    decorator = d;
  }

  cv::Mat RandomCropImageDecorator::Decorate(const cv::Mat& ori){
    cv::Mat img = decorator->Decorate(ori);
    int oriWidth = img.size().width;
    int oriHeight = img.size().height;
    int cropSize = rand() % (maxCropsize - minCropsize + 1);
    cropSize += minCropsize;
    int leftUpperW = rand() % (oriWidth - cropSize + 1);
    int leftUpperH = rand() % (oriHeight - cropSize + 1);
    cropImage(img, img, leftUpperW, leftUpperH, cropSize, cropSize);
    return img;
  }

  LuminanceContrastVariationImageDecorator::LuminanceContrastVariationImageDecorator(
      const shared_ptr<ImageDecorator>& d, float l, float c) :
      luminance_vary(l), contrast_vary(c){
      decorator = d;
    }

  cv::Mat LuminanceContrastVariationImageDecorator::Decorate(const cv::Mat& ori){
    cv::Mat img = decorator->Decorate(ori);
    float luminance;
    float contrast;
    if (luminance_vary != 0){
      caffe_vRngGaussian<float>(1, &luminance, 0, luminance_vary);
    } else {
      luminance = 0;
    }
    if (contrast_vary != 0){
      caffe_vRngUniform<float>(1, &contrast, -contrast_vary, contrast_vary);
    } else {
      contrast = 0;
    }

    // Prepare the lookup table for luminance and contrast adjustment.
    // We assume the input range is [0, 255].
    float mapping[256]; 
    ConstructLookUp(mapping, luminance, contrast);

    int width = img.size().width;
    int height = img.size().height;
    int channels = img.channels();

    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          // We only modify the luminance and contrast of RGB channel
          if (c < 3){
            img.at<cv::Vec3b>(h,w)[c] = mapping[img.at<cv::Vec3b>(h,w)[c]];
          }
        }
      }
    }
    return img;
  }

  // Construct the lookup table for luminance and contrast adjustment.
  // Currently, we assume the mean for natural image is 120. Change it if needed.
  void LuminanceContrastVariationImageDecorator::ConstructLookUp(float* mapping,
  	const float& luminance, const float& contrast){
    float cv = contrast < 0 ? contrast : (1 / (1 - contrast) - 1);
    for (int i = 0; i <= 255; ++i){
      mapping[i] = (i + luminance - 120) * (1 + cv) + 120;
      if (mapping[i] < 0){
        mapping[i] = 0;
      }
      if (mapping[i] > 255){
        mapping[i] = 255;
      }
    }
  }

  BlurImageDecorator::BlurImageDecorator(const shared_ptr<ImageDecorator>& d, 
      float r, float s) : range(r), sigma(s){
      decorator = d;
    }

  cv::Mat BlurImageDecorator::Decorate(const cv::Mat& ori){
    cv::Mat img = decorator->Decorate(ori);
    blurImageBilateral(img, img, range, sigma, sigma);
    return img;
  }

  
} // namespace caffe
