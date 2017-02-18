This MATLAB code implements the SRDCF tracker [1].

Installation:
If the pre-compiled mexfiles do not work, the provided compilemex_[linux|windows] should compile them from the provided source code. Alternatively, you can try to modify them for your system.

Instructions;
* The "demo.m" script runs the tracker on the provided "Couple" sequence.
* The "run_SRDCF.m" function can be directly integrated to the Online Tracking Benchmark (OTB).
* The "run_SRDCF.m" contains the default parameters used to produce the results reported in the paper [1].

Contact:
Martin Danelljan
martin.danelljan@liu.se

Third party code used in the implementation of this tracker is:
* Piotrs image processing toolbox [2]
* mtimesx [3]
* opencv [4]
* lightspeed toolbox [5]


[1] Martin Danelljan, Gustav Häger, Fahad Shahbaz Khan and Michael Felsberg.
	Learning Spatially Regularized Correlation Filters for Visual Tracking.
	In Proceedings of the International Conference in Computer Vision (ICCV), 2015. 

[2] Piotr Dollár.
    "Piotr’s Image and Video Matlab Toolbox (PMT)."
    http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html.

[3] http://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support

[4] http://opencv.org/

[5] http://research.microsoft.com/en-us/um/people/minka/software/lightspeed/
