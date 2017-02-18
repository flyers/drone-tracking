// Copyright 2013 Naiyan Wang

#ifndef CAFFE_UTIL_IMAGE_TRANSFORMATION_H_
#define CAFFE_UTIL_IMAGE_TRANSFORMATION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
using namespace cv;
namespace caffe{
	Mat toMat(const std::string& data, int c, int oriH, int oriW, int h, int w, int h_off = 0, int w_off = 0);
	void resizeImage(const Mat& src, Mat& dst, int w, int h);
	void cropImage(const Mat& src, Mat& dst, int x, int y, int w, int h);
	void rotateImage(const Mat& src, Mat& dst, float angle);
	void flipImage(const Mat& src, Mat& dst);
	void blurImageBilateral(const Mat& src, Mat& dst, float w, float sigma1, float sigma2);
} // namespace caffe

#endif
