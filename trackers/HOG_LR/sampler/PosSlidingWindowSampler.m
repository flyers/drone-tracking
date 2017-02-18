function [tmpl] = PosSlidingWindowSampler(initTmpl, opt)

slidingH        = opt.Sampler.PosSlidingWindowSampler.PosSlidingH;
slidingW        = opt.Sampler.PosSlidingWindowSampler.PosSlidingW;

hVec            = (0:slidingH) - round(slidingH/2);
wVec            = (0:slidingW) - round(slidingW/2);
num             = length(hVec) * length(wVec);
[wMat, hMat]    = meshgrid(wVec, hVec);

tmpl = repmat(initTmpl, [num, 1]);
tmpl(:, 1) = tmpl(:, 1) + wMat(:);
tmpl(:, 2) = tmpl(:, 2) + hMat(:);

tmpl = [initTmpl; tmpl;];

