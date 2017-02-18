function [tmpl] = RadiusSlidingWindowMotionModel(initTmpl, initConf, opt)

radius          = opt.MotionModel.RadiusSlidingWindowMotionModel.radius;
stride          = opt.MotionModel.RadiusSlidingWindowMotionModel.stride;

r2 = radius^2;

hVec            = -radius:stride:radius;
wVec            = -radius:stride:radius;
[wMat, hMat]    = meshgrid(wVec, hVec);

wMat = wMat(:); hMat = hMat(:);
idx = (wMat.^2 + hMat.^2 <= r2);
wMat = wMat(idx); hMat = hMat(idx);
num = sum(idx);


[~, i] = max(initConf);
tempTmpl = repmat(initTmpl(i, :), [num, 1]);
tempTmpl(:, 1) = tempTmpl(:, 1) + wMat;
tempTmpl(:, 2) = tempTmpl(:, 2) + hMat;
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

