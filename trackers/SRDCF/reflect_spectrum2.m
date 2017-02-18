% Do a reflection in the fourier domain for a 2-dimensional signal

function xf_reflected = reflect_spectrum2(xf)
% use fliplr and flipud to replace the flip function or the rot90 func
% xf_reflected = circshift(flip(flip(xf, 1), 2), [1 1 0]);
xf_reflected = circshift(rot90(xf, 2), [1 1 0]);