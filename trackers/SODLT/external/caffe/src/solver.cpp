// Copyright Yangqing Jia 2013

#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : param_(param), net_(), test_net_() {
  // Scaffolding code
  LOG(INFO) << "In Solver Constructor";
  Caffe::set_mode(Caffe::Brew(param_.solver_mode()));
  if (param_.solver_mode()){
    if (param_.has_device_id()) {
      Caffe::SetDevice(param_.device_id());
      LOG(INFO) << "Setting Device ID to " << param_.device_id();
    } else {
      Caffe::SetDevice(0);
      LOG(INFO) << "Seeting Device ID to default 0.";
    }
  }
  Caffe::set_phase(Caffe::TRAIN);
  NetParameter train_net_param;
  ReadProtoFromTextFile(param_.train_net(), &train_net_param);
  LOG(INFO) << "Creating training net.";
  net_.reset(new Net<Dtype>(train_net_param));
  iter_ = 0;
  if (param_.has_test_net()) {
    LOG(INFO) << "Creating testing net.";
    NetParameter test_net_param;
    ReadProtoFromTextFile(param_.test_net(), &test_net_param);
    test_net_.reset(new Net<Dtype>(test_net_param));
    CHECK_GT(param_.test_iter(), 0);
    CHECK_GT(param_.test_interval(), 0);
  }
  LOG(INFO) << "Solver scaffolding done.";
}


template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  PreSolve();
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  while (iter_++ < param_.max_iter()) {
    Dtype loss = DoUpdate();

    if (param_.display() && iter_ % param_.display() == 0) {
      LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
      // We need to set phase to test before running.
      Caffe::set_phase(Caffe::VAL);
      Test();
      Caffe::set_phase(Caffe::TRAIN);
    }
    // Check if we need to do snapshot
    if (param_.snapshot() && iter_ % param_.snapshot() == 0) {
      Snapshot();
    }
  }
  // After the optimization is done, always do a snapshot.
  iter_--;
  Snapshot();
  LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void Solver<Dtype>::Test() {
  LOG(INFO) << "Iteration " << iter_ << ", Testing net";
  NetParameter net_param;
  net_->ToProto(&net_param);
  CHECK_NOTNULL(test_net_.get())->CopyTrainedLayersFrom(net_param);
  vector<Dtype> test_score;
  vector<Blob<Dtype>*> bottom_vec;
  for (int i = 0; i < param_.test_iter(); ++i) {
    //LOG(INFO)<<"test iteration "<<i;
    const vector<Blob<Dtype>*>& result =
        test_net_->Forward(bottom_vec);
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  for (int i = 0; i < test_score.size(); ++i) {
    LOG(INFO) << "Test score #" << i << ": "
        << test_score[i] / param_.test_iter();
  }
}


template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  net_->ToProto(&net_param, param_.snapshot_diff());
  string filename(param_.snapshot_prefix());
  char iter_str_buffer[20];
  sprintf(iter_str_buffer, "_iter_%d", iter_);
  filename += iter_str_buffer;
  LOG(INFO) << "Snapshotting to " << filename;
  WriteProtoToBinaryFile(net_param, filename.c_str());
  SolverState state;
  SnapshotSolverState(&state);
  state.set_iter(iter_);
  state.set_learned_net(filename);
  filename += ".solverstate";
  LOG(INFO) << "Snapshotting solver state to " << filename;
  WriteProtoToBinaryFile(state, filename.c_str());
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  SolverState state;
  NetParameter net_param;
  ReadProtoFromBinaryFile(state_file, &state);
  if (state.has_learned_net()) {
    ReadProtoFromBinaryFile(state.learned_net().c_str(), &net_param);
    net_->CopyTrainedLayersFrom(net_param);
  }
  iter_ = state.iter();
  RestoreSolverState(state);
}


// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
// where base_lr, gamma, step and power are defined in the solver parameter
// protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype Solver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    int current_step = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), current_step);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "nesterov") {
    rate = Dtype(1) / (1 / this->param_.base_lr() +
       this->param_.gamma() * pow(this->iter_, this->param_.power()));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  
  return rate;
}


template <typename Dtype>
void MomentumSolver<Dtype>::PreSolve() {
  // Initialize the history
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  accumulate_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const Blob<Dtype>* net_param = net_params[i].get();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
    accumulate_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
  }
}


template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  vector<float>& net_params_lr = this->net_->params_lr();
  vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  // get the learning rate
  Dtype rate = this->GetLearningRate();
  
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  Dtype momentum = this->param_.momentum();
  Dtype weight_decay = this->param_.weight_decay();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
      caffe_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->cpu_diff(), momentum,
          this->history_[param_id]->mutable_cpu_data());
      if (local_decay) {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay * local_rate,
            net_params[param_id]->cpu_data(),
            this->history_[param_id]->mutable_cpu_data());
      }
      // copy
      caffe_copy(net_params[param_id]->count(),
          this->history_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->gpu_diff(), momentum,
          this->history_[param_id]->mutable_gpu_data());
      if (local_decay) {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay * local_rate,
            net_params[param_id]->gpu_data(),
            this->history_[param_id]->mutable_gpu_data());
      }
      // copy
      caffe_copy(net_params[param_id]->count(),
          this->history_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void MomentumSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  state->clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state->add_history();
    history_[i]->ToProto(history_blob);
  }
}

template <typename Dtype>
void MomentumSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
int Solver<Dtype>::GetUpdateInterval() {
  int ret = 1;
  if (this->param_.has_start_update_interval()){
    ret = this->param_.start_update_interval();
  }
  for (int i = 0; i < this->iter_ / this->param_.double_batch(); ++i){
    ret = ret * 2;
  }
  return ret;
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::DoUpdate() {
  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;
  Dtype loss = this->net_->ForwardBackward(bottom_vec);
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    caffe_gpu_axpy(this->accumulate_[param_id]->count(), Dtype(1),
         net_params[param_id]->gpu_diff(),
         this->accumulate_[param_id]->mutable_gpu_data()
      );
  }
  int updateInterval = this->GetUpdateInterval();
  if (this->iter_ % updateInterval == 0){
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      caffe_copy(net_params[param_id]->count(),
        this->accumulate_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff()
      );
    }
    ComputeUpdateValue();
    this->net_->Update();
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      this->accumulate_[param_id]->Clear();
    }
  }
  return loss;
}

template <typename Dtype>
Dtype NesterovSolver<Dtype>::GetMomentum() {
  Dtype maxmomentum = this->param_.max_momentum();
  Dtype ret = min(maxmomentum, Dtype(1.0 - 
    pow(2.0, -1.0 - log2(this->iter_ / this->param_.momentum_batch() + 1.0))));
  return ret;
}

template <typename Dtype>
Dtype NesterovSolver<Dtype>::DoUpdate() {
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  if (Caffe::mode() != Caffe::GPU) {
    NOT_IMPLEMENTED;
    return Dtype();
  }
  Dtype momentum = GetMomentum();
  vector<float>& net_params_lr = this->net_->params_lr();
  vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  
  int MULTIPLY_BATCH = this->param_.thread_num();
  // Jump the momentum first
  int updateInterval = this->GetUpdateInterval();
  if (updateInterval == 1 || this->iter_ % updateInterval == 1){  
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      caffe_gpu_scal(this->history_[param_id]->count(), momentum, 
        this->history_[param_id]->mutable_gpu_data(), param_id % MULTIPLY_BATCH
      );
      caffe_copy(this->history_[param_id]->count(),
        this->history_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff()
      );
    }
    cudaDeviceSynchronize();
    this->net_->Update(MULTIPLY_BATCH);
  }

  // We delegate the input to net_
  vector<Blob<Dtype>*> bottom_vec;
  Dtype loss = this->net_->ForwardBackward(bottom_vec);
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    caffe_gpu_axpy(this->accumulate_[param_id]->count(), Dtype(1),
         net_params[param_id]->gpu_diff(),
         this->accumulate_[param_id]->mutable_gpu_data(), param_id % MULTIPLY_BATCH
      );
  }
  cudaDeviceSynchronize();
  if (updateInterval == 1 || this->iter_ % updateInterval == 0){  
    Dtype weight_decay = this->param_.weight_decay();
    Dtype rate = this->GetLearningRate();
    if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
      LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate <<
        ", momentum = " << momentum;
    }
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      // Add this to compensate large batch size effect.
      // Naiyan added on Apr.01.14
      caffe_gpu_scal(net_params[param_id]->count(),
          Dtype(1) / updateInterval,
          this->accumulate_[param_id]->mutable_gpu_data(), param_id % MULTIPLY_BATCH  
        );

      if (local_decay){
        caffe_gpu_axpby(net_params[param_id]->count(),
          local_decay * local_rate,
          net_params[param_id]->gpu_data(),
          local_rate,
          this->accumulate_[param_id]->mutable_gpu_data(),  param_id % MULTIPLY_BATCH
        );
      } else {
        caffe_gpu_scal(net_params[param_id]->count(),
          local_rate,
          this->accumulate_[param_id]->mutable_gpu_data(),  param_id % MULTIPLY_BATCH
        );
      }
      
      caffe_copy(net_params[param_id]->count(),
        this->accumulate_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff()
      );

      caffe_gpu_axpy(this->history_[param_id]->count(), Dtype(1), 
        this->accumulate_[param_id]->gpu_data(),
        this->history_[param_id]->mutable_gpu_data(),  param_id % MULTIPLY_BATCH
      );

      this->accumulate_[param_id]->Clear();
    }
    this->net_->Update(MULTIPLY_BATCH);
  }
  return loss;
}

INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(MomentumSolver);
INSTANTIATE_CLASS(SGDSolver);
INSTANTIATE_CLASS(NesterovSolver);
}  // namespace caffe
