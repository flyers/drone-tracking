function [data, opt] = HogExtractor(im, tmpl, opt)
sz = opt.FeatureExtractor.tmplsize;
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


% By Ross fhog
binSize         = 8;
nOrients        = 9;
featureLength   = prod([sz(1)/binSize, sz(2)/binSize, nOrients*3+5]);
features = zeros(featureLength, size(tmpl, 1));

for i = 1:size(tmpl, 1)
    midW    = tmpl(i, 1);
    midH    = tmpl(i, 2);
    w       = tmpl(i, 3);
    h       = tmpl(i, 4);

    tempIm = im(round(midH-h/2) : round(midH+h/2),...
                round(midW-w/2) : round(midW+w/2), :);
%     tempIm = imresize(tempIm, sz);
    tempIm = mexResize(tempIm, sz, 'auto');
%     hogFeatures = extractHOGFeatures(tempIm);
    hogFeatures = fhog(single(tempIm*255), binSize);
    features(:, i) = hogFeatures(:) / norm(hogFeatures(:));
end


data.feat = features;