// Copyright 2013 Yangqing Jia

#ifndef CAFFE_SOLVER_FACTORY_HPP_
#define CAFFE_SOLVER_FACTORY_HPP_

#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp" 

namespace caffe {


// A function to get a specific solver from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Solver<Dtype>* GetSolver(const SolverParameter& param) {
  const std::string& type = param.type();
  if (type == "SGD") {
    return new SGDSolver<Dtype>(param);
  } else if (type == "Nesterov") {
    return new NesterovSolver<Dtype>(param);
  } else {
    LOG(FATAL) << "Unknown solver name: " << type;
  }
  // just to suppress old compiler warnings.
  return (Solver<Dtype>*)(NULL);
}

template Solver<float>* GetSolver(const SolverParameter& param);
template Solver<double>* GetSolver(const SolverParameter& param);

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_HPP_
