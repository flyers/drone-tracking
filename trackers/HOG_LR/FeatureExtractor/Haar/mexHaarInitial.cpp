#include "mex.h"

//static const int kHaarFeatureCount = 588;
static const int kHaarFeatureCount = 192;

double *hMin; double *height; double *wMin; double *width; double *weight;
double *factor; double *area;
void setMatrix(double iwMin, double ihMin, double iwidth, 
		double iheight, double iweight, int ifeature, int idx){
	int pos = ifeature + kHaarFeatureCount * idx;
	wMin[pos] 		= iwMin;
	hMin[pos] 		= ihMin;
	width[pos] 		= iwidth;
	height[pos]		= iheight;
	weight[pos]		= iweight;
}

void setTemplate(int it, int ifeature, double iwMin, double ihMin, double iwidth, double iheight){
	if (it == 0){
		setMatrix(iwMin, ihMin, 				iwidth, iheight / 2, 1.f, 	ifeature, 0);
		setMatrix(iwMin, ihMin + iheight / 2, 	iwidth, iheight / 2, -1.f, 	ifeature, 1);
		factor[ifeature] = 1.f/2;
	}
	else if (it == 1){
		setMatrix(iwMin, 				ihMin, iwidth / 2, iheight, 1.f, 	ifeature, 0);
		setMatrix(iwMin + iwidth / 2, 	ihMin, iwidth / 2, iheight, -1.f, 	ifeature, 1);
		factor[ifeature] = 1.f/2;
	}
	else if (it == 2){
		setMatrix(iwMin, 				ihMin, iwidth / 3, iheight, 1.f, 	ifeature, 0);
		setMatrix(iwMin + iwidth / 3, 	ihMin, iwidth / 3, iheight, -2.f, 	ifeature, 1);
		setMatrix(iwMin + 2 * iwidth/3,	ihMin, iwidth / 3, iheight, 1.f, 	ifeature, 2);
		factor[ifeature] = 2.f/3;
	}
	else if(it == 3){
		setMatrix(iwMin, ihMin, 					iwidth, iheight / 3, 1.f, 	ifeature, 0);
		setMatrix(iwMin, ihMin + iheight / 3, 		iwidth, iheight / 3, -2.f, 	ifeature, 1);
		setMatrix(iwMin, ihMin + 2 * iheight / 3,	iwidth, iheight / 3, 1.f, 	ifeature, 2);
		factor[ifeature] = 2.f/3;
	}
	else if(it == 4){
		setMatrix(iwMin, 				ihMin, 					iwidth / 2, iheight / 2, 1.f, 	ifeature, 0);
		setMatrix(iwMin + iwidth / 2, 	ihMin + iheight / 2, 	iwidth / 2, iheight / 2, 1.f, 	ifeature, 1);
		setMatrix(iwMin, 				ihMin + iheight / 2, 	iwidth / 2, iheight / 2, -1.f, 	ifeature, 2);
		setMatrix(iwMin + iwidth / 2, 	ihMin, 					iwidth / 2, iheight / 2, -1.f, 	ifeature, 3);
		factor[ifeature] = 1.f/2;
	}
	else if(it == 5){
		setMatrix(iwMin, 				ihMin, 					iwidth, 	iheight, 	 1.f, 	ifeature, 0);
		setMatrix(iwMin + iwidth / 4, 	ihMin + iheight / 4, 	iwidth / 2, iheight / 2, -4.f, 	ifeature, 1);
		factor[ifeature] = 3.f/4;
	}		
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]){         
    plhs[0] = mxCreateDoubleMatrix(kHaarFeatureCount, 4, mxREAL); 
    plhs[1] = mxCreateDoubleMatrix(kHaarFeatureCount, 4, mxREAL); 
    plhs[2] = mxCreateDoubleMatrix(kHaarFeatureCount, 4, mxREAL); 
    plhs[3] = mxCreateDoubleMatrix(kHaarFeatureCount, 4, mxREAL); 
    plhs[4] = mxCreateDoubleMatrix(kHaarFeatureCount, 4, mxREAL); 

    hMin = mxGetPr(plhs[0]);
    height = mxGetPr(plhs[1]);    
    wMin = mxGetPr(plhs[2]);    
    width = mxGetPr(plhs[3]);    
    weight = mxGetPr(plhs[4]);
    
    plhs[5] = mxCreateDoubleMatrix(kHaarFeatureCount, 1, mxREAL); 
    plhs[6] = mxCreateDoubleMatrix(kHaarFeatureCount, 1, mxREAL); 
    factor = mxGetPr(plhs[5]);
    area = mxGetPr(plhs[6]);
    
	double x[] = {0.2f, 0.4f, 0.6f, 0.8f};
	double y[] = {0.2f, 0.4f, 0.6f, 0.8f};
	int lengthX = 4; int lengthY = 4;
	//double x[] = {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
	//double y[] = {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
	//int lengthX = 7; int lengthY = 7;
	double s[] = {0.2f, 0.4f};
	int ifeature = 0;
	for (int iy = 0; iy < lengthY; ++iy)
	{
		for (int ix = 0; ix < lengthX; ++ix)
		{
			for (int is = 0; is < 2; ++is)
			{
				double ihMin = x[ix]-s[is]/2;
				double iwMin = y[iy]-s[is]/2;
				double iheight = s[is];
				double iwidth = s[is];

				for (int it = 0; it < 6; ++it){
					area[ifeature] = iheight * iwidth;
					setTemplate(it, ifeature, iwMin, ihMin, iwidth, iheight);
					ifeature++;		
				}
			}
		}
	}
}