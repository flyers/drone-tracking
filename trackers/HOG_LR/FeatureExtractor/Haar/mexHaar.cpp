#include "mex.h"
#include <cmath>

//static const int kHaarFeatureCount = 588;
static const int kHaarFeatureCount = 192;

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]){ 
	double *hMin = mxGetPr(prhs[0]); 
    double *height = mxGetPr(prhs[1]);
    double *wMin = mxGetPr(prhs[2]); 
    double *width = mxGetPr(prhs[3]);
    double *weight = mxGetPr(prhs[4]);
    double *factor = mxGetPr(prhs[5]);
    double *area = mxGetPr(prhs[6]);

    double *tmpl = mxGetPr(prhs[7]);
    double *integralIm = mxGetPr(prhs[8]);
    
    int numTmpl = mxGetM(prhs[7]);
    int imh = mxGetM(prhs[8]);
    //int imw = mxGetN(prhs[7]);

    
    plhs[0] = mxCreateDoubleMatrix(kHaarFeatureCount, numTmpl, mxREAL); 
    double *output = mxGetPr(plhs[0]);
    
    for (int iTmpl = 0; iTmpl < numTmpl; iTmpl++){
        double iwMin = tmpl[iTmpl]             - tmpl[iTmpl + numTmpl * 2] / 2;
        double ihMin = tmpl[iTmpl + numTmpl]   - tmpl[iTmpl + numTmpl * 3] / 2;
        double iwidth = tmpl[iTmpl + numTmpl * 2];
        double iheight = tmpl[iTmpl + numTmpl * 3];

        for (int j = 0; j < kHaarFeatureCount; j++){
            for (int i = 0; i < 4; ++i){
                int pos = j + kHaarFeatureCount * i;
                //mexPrintf("Pos: %d\n", pos);
                if (weight[pos] != 0){
                    int ht = round(ihMin + iheight * hMin[pos] - 0.5f);
                    int wl = round(iwMin + iwidth  * wMin[pos] - 0.5f);
                    int hb = round(ihMin + iheight * (hMin[pos] + height[pos]) - 0.5f);
                    int wr = round(iwMin + iwidth  * (wMin[pos] + width [pos]) - 0.5f);

                    //mexPrintf("%d, %d, %d, %d, %d, %d\n", ht, wl, hb, wr, imh, imw);
                    output[j + kHaarFeatureCount * iTmpl] += weight[pos] * 
                            (integralIm[ht + wl * imh] + integralIm[hb + wr * imh] 
                            - integralIm[ht + wr * imh] - integralIm[hb + wl * imh]);
                }
            }
            output[j + kHaarFeatureCount * iTmpl] /= (factor[j] * area[j] * iwidth * iheight);
        }
        //mexPrintf("********iTmpl: %d\n", iTmpl);
    }
    //mexPrintf("Feature Extraction done!\n");
}