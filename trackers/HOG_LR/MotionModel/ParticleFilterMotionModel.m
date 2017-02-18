function [tmpl] = ParticleFilterMotionModel(initTmpl, initConf, opt)

varTmpl         = opt.MotionModel.ParticleFilterMotionModel.affsig;
N               = opt.MotionModel.ParticleFilterMotionModel.N;
szH              = opt.FeatureExtractor.tmplsize(1);
% initConf        = initConf / sum(initConf);

initConf = initConf - min(initConf);
initConf = exp(double(initConf) ./opt.condenssig)';
initConf = initConf ./ sum(initConf);
[~, i] = max(initConf);

if size(initTmpl, 1) == 1
    tmpl            = repmat(initTmpl, [N, 1]);
else
    N               = size(initTmpl, 1);
    cumconf         = cumsum(initConf);
    idx             = floor(sum(repmat(rand(1,N),[N,1]) > repmat(cumconf,[1, N])))+1;
    tmpl            = initTmpl(idx, :);
end

tmpl(:, 4)      = tmpl(:, 4) ./ tmpl(:, 3);
tmpl(:, 3)      = tmpl(:, 3) / szH;
tmpl            = tmpl + randn(N, 4).*repmat(varTmpl,[N, 1]);
tmpl(:, 3)      = tmpl(:, 3) * szH;
tmpl(:, 4)      = tmpl(:, 3) .* tmpl(:, 4);

rndIdx  = randperm(size(tmpl, 1));
tmpl    = tmpl(rndIdx(1:end - 1), :);
tmpl    = [initTmpl(i, :); tmpl]; % reserve the best one from the last frame

idx = (tmpl(:, 3) > 3 & tmpl(:, 4) > 3);
tmpl = tmpl(idx, :);



