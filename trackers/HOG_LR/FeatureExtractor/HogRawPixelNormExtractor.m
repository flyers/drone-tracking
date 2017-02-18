function [data, opt] = HogRawPixelNormExtractor(im, tmpl, opt)

[dataHog] = HogExtractor(im, tmpl, opt);
temp = opt; 
temp.FeatureExtractor.tmplsize = temp.FeatureExtractor.tmplsize / 2;
[dataRaw] = RawPixelExtractor(im, tmpl, temp);

data.feat = [dataHog.feat; dataRaw.feat];
dataNorm = sqrt(sum(data.feat .* data.feat));
data.feat = bsxfun(@rdivide, data.feat, dataNorm);
data.tmpl = dataHog.tmpl;
