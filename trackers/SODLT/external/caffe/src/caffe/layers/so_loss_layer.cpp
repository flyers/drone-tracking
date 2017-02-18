// Copyright 2013 Naiyan Wang

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
using namespace std;

namespace caffe {

	template <typename Dtype>
	void StructureOutputLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
	      vector<Blob<Dtype>*>* top) {
	  CHECK_EQ(bottom.size(), 2) << "Box Layer takes two blobs as input.";
	  CHECK_EQ(top->size(), 0) << "Box Layer takes no blob as output.";
	  NUM_ = bottom[0]->num();
	  HEIGHT_ = sqrt((Dtype)bottom[0]->channels());
	  WIDTH_ = sqrt((Dtype)bottom[0]->channels());
	  losses.Reshape(NUM_, 1, 1, 1);
	};

	template <typename Dtype>
	void StructureOutputLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	    vector<Blob<Dtype>*>* top) {
	   NOT_IMPLEMENTED;
	}

	template <typename Dtype>
	Dtype StructureOutputLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	    const bool propagate_down, vector<Blob<Dtype>*>* bottom){
	    NOT_IMPLEMENTED;
      return (Dtype)0.0;
	}
  INSTANTIATE_CLASS(StructureOutputLossLayer);
}