function [data, opt] = HaarExtractor(im, tmpl, opt)

im = rgb2gray(im);
data.tmpl = tmpl;   

% pad
minW = min(round(tmpl(:, 1) - tmpl(:, 3) / 2)) - 1;
maxW = max(round(tmpl(:, 1) + tmpl(:, 3) / 2)) + 1;
minH = min(round(tmpl(:, 2) - tmpl(:, 4) / 2)) - 1;
maxH = max(round(tmpl(:, 2) + tmpl(:, 4) / 2)) + 1;
[h, w, c] = size(im);
if (minW < 1)
    im_new = zeros(h, w + abs(minW) + 1, c);
    im_new(:, abs(minW) + 2:end, :) = im;
    im = im_new;
    tmpl(:, 1) = tmpl(:, 1) + abs(minW) + 1; 
end
if (maxW > w)
    im_new = zeros(h, size(im, 2) + maxW - w, c);
    im_new(:, 1:size(im, 2), :) = im;
    im = im_new;
end
if (minH < 1)
    im_new = zeros(h + abs(minH) + 1, size(im, 2), c);
    im_new(abs(minH) + 2:end, :, :) = im;
    im = im_new;
    tmpl(:, 2) = tmpl(:, 2) + abs(minH) + 1;
end
if (maxH > h)
    im_new = zeros(size(im, 1) + maxH - h, size(im, 2), c);
    im_new(1:size(im, 1), :, :) = im;
    im = im_new;
end

if ~isfield(opt, 'HaarOpt')
    [opt.HaarOpt.hMin, ...
        opt.HaarOpt.height, ...
        opt.HaarOpt.wMin, ...
        opt.HaarOpt.width, ...
        opt.HaarOpt.weight, ...
        opt.HaarOpt.factor, ...
        opt.HaarOpt.area] = mexHaarInitial(); 
end

integralIm = integralImage(im);
features = mexHaar(opt.HaarOpt.hMin, ...
        opt.HaarOpt.height, ...
        opt.HaarOpt.wMin, ...
        opt.HaarOpt.width, ...
        opt.HaarOpt.weight, ...
        opt.HaarOpt.factor, ...
        opt.HaarOpt.area, ...
        tmpl, integralIm);

div = sqrt(sum(features.*features));
div(div < 1e-6) = 1;
data.feat = bsxfun(@rdivide, features, div);
data.tmpl = tmpl;