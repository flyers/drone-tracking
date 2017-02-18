function [tmpl] = NegSlidingWindowSampler(initTmpl, opt)

slidingH        = opt.Sampler.NegSlidingWindowSampler.NegSlidingH;
slidingW        = opt.Sampler.NegSlidingWindowSampler.NegSlidingW;
stride          = opt.Sampler.NegSlidingWindowSampler.NegStride;
exclude         = opt.Sampler.NegSlidingWindowSampler.excludeNegRatio;

hVec            = (0:stride:slidingH) - round(slidingH/2);
wVec            = (0:stride:slidingW) - round(slidingW/2);
num             = length(hVec) * length(wVec);
[wMat, hMat]    = meshgrid(wVec, hVec);

tmpl            = zeros(num * size(initTmpl, 1), 4);
idx             = zeros(num * size(initTmpl, 1), 1);

for i = 1:size(initTmpl, 1)
    tempTmpl = repmat(initTmpl(i, :), [num, 1]);
    tempTmpl(:, 1) = tempTmpl(:, 1) + wMat(:);
    tempTmpl(:, 2) = tempTmpl(:, 2) + hMat(:);
    
    idx((i-1)*num+1 : i*num)        = abs(hMat(:)) > tempTmpl(i, 3) * exclude | wMat(:) > tempTmpl(i, 4) * exclude;
    tmpl((i-1)*num+1 : i*num, :)    = tempTmpl;
end

tmpl = tmpl((idx == 1), :);


