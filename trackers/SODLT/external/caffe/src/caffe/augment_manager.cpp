#include "caffe/augment_manager.hpp"
#include "caffe/common.hpp"
#include <vector>
#include <utility>

using namespace std;
namespace caffe{
	ImageAugmentationManager::ImageAugmentationManager(){
	
	}
	ImageAugmentationManager::~ImageAugmentationManager(){
	
	}
	void ImageAugmentationManager::ParseConfig(const ImageAugmentationParam &params,
		Caffe::Phase phase, int outputSize){
		shared_ptr<ImageDecorator> decorator;
		decorator.reset(new OriginalImageDecorator());
		vector<int> resolves;
		switch (phase){
			case Caffe::TRAIN:
				if (params.has_min_cropsize() && !params.has_max_cropsize() ||
					!params.has_min_cropsize() && params.has_max_cropsize()){
					LOG(FATAL) << "min_cropsize and max_cropsize must occur at the same time.";
				}
				int minCropsize, maxCropsize;
				if (params.has_min_cropsize() && params.has_max_cropsize()){
					minCropsize = params.min_cropsize();
					maxCropsize = params.max_cropsize();
					LOG(INFO) << "Training takes random scale, min: " << minCropsize << ", max: " << maxCropsize;
					decorator.reset(new RandomCropImageDecorator(decorator, minCropsize, maxCropsize));
				}
				break;
			case Caffe::VAL:
			case Caffe::TEST:
				CHECK(params.has_side() && params.has_pos()) << "size and pos must be specified in test net.";
				CHECK_EQ(params.resolve_size_size(), 1) << "Only one resolve size needed in each augment configuration.";
				// We only care the first resolve size input.
				decorator.reset(new CropImageDecorator(decorator, params.side()));
				decorator.reset(new CropImageDecorator(decorator, params.pos(), params.resolve_size(0)));
				break;
			default:
				LOG(FATAL) << "Unknown phase.";
		}
		
		decorator.reset(new ResizeImageDecorator(decorator, outputSize, outputSize));
		LOG(INFO) << "Training input size is " << outputSize;
		if (params.mirror()){
			decorator.reset(new FlipImageDecorator(decorator, phase != Caffe::TRAIN));
			if (phase != Caffe::TEST){
				LOG(INFO) << "Uses random flip.";
			}
		}
		if (phase == Caffe::TRAIN && params.has_luminance_vary() && params.has_contrast_vary()){
			decorator.reset(new LuminanceContrastVariationImageDecorator(decorator, params.luminance_vary(),
				params.contrast_vary()));
			if (phase != Caffe::TEST){
				LOG(INFO) << "Uses luminance and contrast variation: " <<
					params.luminance_vary() << " " << params.contrast_vary();
			}
		}

		if (params.has_blur_range() && params.has_blur_sigma()){
			decorator.reset(new BlurImageDecorator(decorator, params.blur_range(), params.blur_sigma()));
			if (phase != Caffe::TEST){
				LOG(INFO) << "Uses blur variation: " <<
					params.blur_range() << " " << params.blur_sigma();
			}
		}

		if (phase == Caffe::TRAIN || phase == Caffe::VAL){
			trainDecorators.push_back(decorator);
		} else {
			testDecorators.push_back(decorator);
		}
	}

	cv::Mat ImageAugmentationManager::Decorate(const cv::Mat& img, Caffe::Phase phase, int id){
		vector<shared_ptr<ImageDecorator> >& decorators = (phase == Caffe::TEST ? testDecorators : trainDecorators);
		CHECK_LT(id, decorators.size());
		//LOG(INFO) << "---" << img.cols << " " << img.rows;
		return decorators[id]->Decorate(img);
	}

	int ImageAugmentationManager::GetSize(Caffe::Phase phase){
		vector<shared_ptr<ImageDecorator> >& decorators = (phase == Caffe::TEST ? testDecorators : trainDecorators);
		return decorators.size();
	}
}