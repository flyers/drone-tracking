function [data, opt] = GrayRawPixelExtractor(im, tmpl, opt)
sz = opt.FeatureExtractor.tmplsize;
if (ndims(im) == 3)
    im = rgb2gray(im);
end
data.tmpl = tmpl;   

features = zeros(prod(sz)*size(im, 3), size(tmpl, 1));

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
    
for i = 1:size(tmpl, 1)
    midW    = tmpl(i, 1);
    midH    = tmpl(i, 2);
    w       = tmpl(i, 3);
    h       = tmpl(i, 4);

    tempIm = im(round(midH-h/2) : round(midH+h/2),...
                round(midW-w/2) : round(midW+w/2), :);
%     tempIm = imresize(tempIm, sz);
    tempIm = mexResize(tempIm, sz, 'auto');
    features(:, i) = tempIm(:);
    if (norm(features(:, i)) > 1e-6)
        features(:, i) = features(:, i) / norm(features(:, i));
    end
end
% features = features - 0.5;
    
data.feat = features;