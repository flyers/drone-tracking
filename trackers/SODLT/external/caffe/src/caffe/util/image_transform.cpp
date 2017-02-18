#include "caffe/util/image_transform.hpp"
#include "caffe/common.hpp"
using namespace cv;
namespace caffe{
	Mat toMat(const std::string& data, int channels, int oriHeight, int oriWidth,
      int height, int width, int h_off, int w_off){
		Mat img = Mat::zeros(height, width, CV_8UC3);
		for(int c = 0;c < channels; ++c){
			for(int h = 0; h < height; ++h){
				for(int w = 0; w < width;++w){
					img.at<cv::Vec3b>(h, w)[c] = data[(c * oriHeight + h + h_off) * oriWidth + w + w_off];
				}
			}
    }
    return img;
	}

	void cropImage(const Mat& src, Mat& dst, int x, int y, int w, int h){
    
		dst = src(Rect(x, y, w, h));
	}

	void flipImage(const Mat& src, Mat& dst){
		flip(src, dst, 1);
	}

	void resizeImage(const Mat& src, Mat& dst, int w, int h){
		if (h == src.rows && w == src.cols){
			dst = src;
      return;
		}
		resize(src, dst, Size(w, h), 0, 0, CV_INTER_CUBIC);
	}

	void rotateImage(const Mat& src, Mat& dst, float angle){
		if (angle == 0){
      dst = src;
      return;
    }
    int len = std::max(src.cols, src.rows);
    cv::Point2f pt(len / 2.0, len / 2.0);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(len, len));
	}

	void blurImageBilateral(const Mat& src, Mat& dst, float w, float sigma1, float sigma2){
		bilateralFilter(src, dst, w, sigma1, sigma2);
	}
} //end of namespace caffe
