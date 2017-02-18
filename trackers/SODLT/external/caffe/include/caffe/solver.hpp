// Copyright Yangqing Jia 2013

#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <vector>
#include <string>
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

using std::string;
using std::vector;
namespace caffe {

template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  virtual ~Solver() {}
  inline Net<Dtype>* net() { return net_.get(); }
  // PreSolve is run before any solving iteration starts, allowing one to
  // put up some scaffold.
  virtual void PreSolve() {}
  // Get the update value for the current iteration.
  virtual Dtype DoUpdate() = 0;
 protected:
  
  
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // The test routine
  void Test();
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;
  //Implement different learning rate scheme.
  Dtype GetLearningRate();
  int GetUpdateInterval();
  SolverParameter param_;
  int iter_;
  shared_ptr<Net<Dtype> > net_;
  shared_ptr<Net<Dtype> > test_net_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};


template <typename Dtype>
class MomentumSolver : public Solver<Dtype> {
 public:
  explicit MomentumSolver(const SolverParameter& param)
      : Solver<Dtype>(param) {}
  virtual ~MomentumSolver() {}

  virtual void PreSolve();
 protected:
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  vector<shared_ptr<Blob<Dtype> > > history_;
  vector<shared_ptr<Blob<Dtype> > > accumulate_;

  DISABLE_COPY_AND_ASSIGN(MomentumSolver);
};

// Implement the plain momentum based SGD solver.
template <typename Dtype>
class SGDSolver : public MomentumSolver<Dtype> {
  public:
    explicit SGDSolver(const SolverParameter& param)
        : MomentumSolver<Dtype>(param) {}
    virtual ~SGDSolver() {}
    virtual Dtype DoUpdate();
  protected:
    void ComputeUpdateValue();
  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

// Implement the Nesterov solver based on the variant in the paper:
// "On the Importance of Momentum and Initialization of Deep Learning"
template <typename Dtype>
class NesterovSolver : public MomentumSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : MomentumSolver<Dtype>(param) {}
  virtual ~NesterovSolver() {}
  virtual Dtype DoUpdate();
 protected:
  Dtype GetMomentum();
  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

// The solver factory function
template <typename Dtype>
Solver<Dtype>* GetSolver(const SolverParameter& param);

}  // namspace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
