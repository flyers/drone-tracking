// Copyright 2014 Naiyan Wang
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <climits>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using namespace std;
using std::max;

namespace caffe {

template <typename Dtype>
void RankingLossLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Ranking loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "Ranking loss Layer takes no output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
}

template <typename Dtype>
Dtype RankingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();

  memset(bottom_diff, 0, (*bottom)[0]->count() * sizeof(Dtype));
  Dtype loss = 0;
  int correct = 0;
  for (int i = 0; i < num; ++i){
    int l = (int)bottom_label[i];
    Dtype target = bottom_data[i * dim + l];
    for (int j = 0; j < dim; ++j){
      Dtype output = bottom_data[i * dim + j];
      if (output > target - 1 && j != l){
        loss += (output - target + 1) * (output - target + 1);
        bottom_diff[i * dim + j] += (output - target + 1) / (num * dim);
        bottom_diff[i * dim + l] += -(output - target + 1) / (num * dim);
      }
    }
  }
  return loss / num;

}
INSTANTIATE_CLASS(RankingLossLayer);
} // namespace caffe
