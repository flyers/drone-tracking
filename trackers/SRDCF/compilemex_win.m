% Run this file to build the needed mex-files on windows

% Build merResize
mex -lopencv_core242 -lopencv_imgproc242 -L./ -I./ mexResize.cpp MxArray.cpp

% Build setnonzeros from the lightspeed matlab toolbox
mex setnonzeros.c

% Build gradientMex from Piotrs toolbox
mex gradientMex.cpp -I./

% Build mtimesx
mtimesx_build