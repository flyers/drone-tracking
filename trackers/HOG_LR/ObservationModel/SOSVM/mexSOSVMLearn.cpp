#include <vector>
#include <cstring>
#include <cstdio>
#include "Config.h"
#include "Kernels.h"
#include "Rect.h"
#include "LaRank.h"
#include "mex.h"

#define CMD_LEN 2048

using namespace std;

static Config* global_conf;
static Kernel* global_kernel;
static LaRank* global_learner;

void exit_with_help ()
{
  mexPrintf(
  "Usage: [label, score] = mexSOSVMLearn(training_rect_matrix, training_instance_matrix, "
          "'task', libsvm_options')\n"
   "options:\n"
   "-c svmC : (default 1)\n"
   "-b budgetSize : (default 0):\n"
   "-t kernel function (default 0):\n"
   "\t0 linear : (default) \n"
   "\t1 gaussian : \n"
   "\t2 intersection : \n"
   "\t3 chi3 : \n"
   "-g gamma : coefficent for polynomial and gaussian kernels (default 1)\n");
}

static void fake_answer(int nlhs, mxArray *plhs[])
{
  int i;
  for(i=0;i<nlhs;i++)
    plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

const char *task_char[] = {"batchTrain", "onlineTrain", "test", "delete"};
vector<string> task_handle(task_char, task_char + 4);

void initialize_config_and_kernel(const mxArray *options){
  // parse libsvm options
  //mexPrintf("ok...\n");
  int budgetSize = 0;
  double C = 1.0;
  int kernel = 0;
  double gaussian_sigma = -1.0;

  char cmd[CMD_LEN];
  char *argv[CMD_LEN/2];
  int argc = 1;
  mxGetString(options, cmd, mxGetN(options) + 1);
  if((argv[argc] = strtok(cmd, " ")) != NULL)
    while((argv[++argc] = strtok(NULL, " ")) != NULL)
      ;

  // parse options
  for (int i = 1; i < argc; i++)
  {
    if (argv[i][0] != '-') break;
    ++i;
    switch (argv[i - 1][1])
    {
      case 'c':
        C = atof (argv[i]);
        mexPrintf("svmC: %f\t", C);
        break;
      case 'b':
        budgetSize = atoi (argv[i]);
        mexPrintf("budgetSize: %d\t", budgetSize);
        break;
      case 't':
        kernel = atoi (argv[i]);
        mexPrintf("kernel: %d\t", kernel);
        break;
      case 'g':
      	gaussian_sigma = atof (argv[i]);
      	break;
      default:
        mexPrintf("Unknown option -%c\n", argv[i-1][1]);
    }
  }
	global_conf = new Config(C, budgetSize, kernel, gaussian_sigma);

	switch (kernel)
	{
		case Config::kKernelTypeLinear:
			global_kernel = new LinearKernel();
			break;
		case Config::kKernelTypeGaussian:
			global_kernel = new GaussianKernel(gaussian_sigma);
			break;
		case Config::kKernelTypeIntersection:
			global_kernel = new IntersectionKernel();
			break;
		case Config::kKernelTypeChi2:
			global_kernel = new Chi2Kernel();
			break;
		default:
			global_kernel = new LinearKernel();
	}
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	if(nrhs > 4 || nrhs < 3) {
    exit_with_help();
    fake_answer(nlhs, plhs);
    return;
  }

  // read training rect matrix
  vector<FloatRect> rects;  
  double* rect_ptr = mxGetPr(prhs[0]);
  int nb_sample = mxGetM(prhs[0]);
  
  rects.resize(nb_sample);
  for (int i = 0; i < nb_sample; i++){
  	rects[i].Set(rect_ptr[i], rect_ptr[i + nb_sample], rect_ptr[i + nb_sample * 2], rect_ptr[i + nb_sample * 3]);
  }

  // read training instance matrix
  if (nb_sample != mxGetM(prhs[1])){
  	mexPrintf("Number of rect does not equal to number of features!\n");
  	fake_answer(nlhs, plhs);
  }
  int nb_feature = mxGetN(prhs[1]);

  double* feature_ptr = mxGetPr(prhs[1]);
  vector<vector<double> > features(nb_sample, vector<double>(nb_feature));
  for (int i = 0; i < nb_sample; i++){
  	for (int j = 0; j < nb_feature; j++){
			features[i][j] = feature_ptr[j * nb_sample + i];
  	}
  }

  // task selection
  char* task = mxArrayToString(prhs[2]);
  if (task_handle[0].compare(task) == 0){ 
		// batch train
          if(mxGetN(prhs[0]) != 4){
            mexPrintf("Rect matrix should be with dimension 4!\n");
            fake_answer(nlhs, plhs);
          } 
          
		if (nlhs != 0){
			mexPrintf("Training has no output!\n");
			fake_answer(nlhs, plhs);
			return;
		}

		if (nrhs == 4){
			initialize_config_and_kernel(prhs[3]);
		} else {
			global_conf = new Config();
			global_kernel = new LinearKernel();
		}

		global_learner = new LaRank(*global_conf, *global_kernel);
		global_learner->Update(features, rects, 0);
	} 
	else if (task_handle[1].compare(task) == 0){ 
		// online train
          if(mxGetN(prhs[0]) != 4){
            mexPrintf("Rect matrix should be with dimension 4!\n");
            fake_answer(nlhs, plhs);
          } 
          
		if (nlhs != 0){
			mexPrintf("Training has no output!\n");
			fake_answer(nlhs, plhs);
			return;
		}

		if (global_learner == NULL || global_kernel == NULL || global_conf == NULL){
			mexPrintf("Classifier Not Initialized!\n");
            fake_answer(nlhs, plhs);
		}

		global_learner->Update(features, rects, 0);
	} 
	else if (task_handle[2].compare(task) == 0){ 
		// test
          if(mxGetN(prhs[0]) != 4){
            mexPrintf("Rect matrix should be with dimension 4!\n");
            fake_answer(nlhs, plhs);
          } 
          
		if (nlhs != 1){
			mexPrintf("Testing should have one output!\n");
			fake_answer(nlhs, plhs);
			return;
		}
		if (global_learner == NULL || global_kernel == NULL || global_conf == NULL){
			mexPrintf("Classifier Not Initialized!\n");
            fake_answer(nlhs, plhs);
		}

        plhs[0] = mxCreateDoubleMatrix(nb_sample, 1, mxREAL);
        double* results = mxGetPr(plhs[0]);

		global_learner->Eval(features, rects, results);
	} else if (task_handle[3].compare(task) == 0){
      // delete and release memory
      delete global_learner;
      delete global_kernel;
      delete global_conf;
      mexPrintf("delete\n");
    } else {
		mexPrintf("Invalid task!\n");
		fake_answer(nlhs, plhs);
	}
	return;
}