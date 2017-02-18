function [tmpl] = SlidingWindowMotionModel(initTmpl, initConf, opt)

slidingH        = opt.MotionModel.SlidingWindowMotionModel.slidingH;
slidingW        = opt.MotionModel.SlidingWindowMotionModel.slidingW;
stride          = opt.MotionModel.SlidingWindowMotionModel.stride;

hVec            = (0:stride:slidingH) - round(slidingH/2);
wVec            = (0:stride:slidingW) - round(slidingW/2);
num             = length(hVec) * length(wVec);
[wMat, hMat]    = meshgrid(wVec, hVec);

[~, i] = max(initConf);
tempTmpl = repmat(initTmpl(i, :), [num, 1]);
tempTmpl(:, 1) = tempTmpl(:, 1) + wMat(:);
tempTmpl(:, 2) = tempTmpl(:, 2) + hMat(:);
tmpl = tempTmpl;

% idx =   round(tmpl(:, 1) - tmpl(:, 3)/2) > 0 & ...
%         round(tmpl(:, 1) + tmpl(:, 3)/2) <= opt.normalWidth &...
%         round(tmpl(:, 2) - tmpl(:, 4)/2) > 0 & ...
%         round(tmpl(:, 2) + tmpl(:, 4)/2) <= opt.normalHeight;
% tmpl = round(tmpl(idx, :));

rndIdx = randperm(size(tmpl, 1));
if ~isfield(opt, 'N')
    N = size(tmpl, 1);
else
    N = opt.N;
end

tmpl    = tmpl(rndIdx(1:N - 1), :);
tmpl    = [initTmpl(i, :); tmpl];


