#ifndef CAFFE_AUGMENT_MANAGER_HPP_
#define CAFFE_AUGMENT_MANAGER_HPP_

#include "caffe/image_decorator.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
	class ImageAugmentationManager{
		public:
			ImageAugmentationManager();
			~ImageAugmentationManager();
			void ParseConfig(const ImageAugmentationParam &params, Caffe::Phase phase,
				int outputSize);
			cv::Mat Decorate(const cv::Mat& img, Caffe::Phase phase, int id = 0);
			int GetSize(Caffe::Phase phase);
		protected:
			vector<shared_ptr<ImageDecorator> > trainDecorators, testDecorators;
	};
} // namespace caffe
#endif