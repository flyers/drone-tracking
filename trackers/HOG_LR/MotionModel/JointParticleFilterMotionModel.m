function [tmplH, tmplTracker, idxH, idxTracker] = JointParticleFilterMotionModel(initH, initTracker, confH, confTracker, opt)
%% tmplTracker M by N by 4


varTmpl         = opt.MotionModel.ParticleFilterMotionModel.affsig;
N               = opt.MotionModel.ParticleFilterMotionModel.N;
M               = opt.H.num;
szH             = opt.FeatureExtractor.tmplsize(1);


% initConf = initConf - min(initConf);
% initConf = exp(double(initConf) ./opt.condenssig)';
% initConf = initConf ./ sum(initConf);
% [~, i] = max(initConf);

if size(initTracker, 1) == 1
    %% for camera motion part
    H 				= initH;
    affparam 		= affparam2geom( [H(1,3), H(2,3), H(1,1), H(1,2), H(2,1), H(2,2)]' );
    projparam 		= H(3, 1:2)';
    tmplH.aff 		= repmat(affparam, [1, M]);
    tmplH.proj 		= repmat(projparam, [1, M]);
    affmat 			= affparam2mat(tmplH.aff);
    tmplH.project 	= [affmat; tmplH.proj; ones(1, M)];
    %% for tracker part
    tmplTracker = repmat(initTracker, [N, 1, M]);
else
    %% for camera motion part
	cumconfH 		= cumsum(confH);
    idxH 			= floor( sum( repmat( rand(1, M), [M, 1] )  > repmat(cumconfH, [1, M]) ) ) + 1;
    tmplH.aff 		= initH.aff(:, idxH);
    tmplH.proj 		= initH.proj(:, idxH);
    tmplH.aff       = tmplH.aff + randn(6, M) .* repmat(opt.H.affsig(:), [1, M]);
    tmplH.proj      = tmplH.proj + randn(2, M) .* repmat(opt.H.projsig(:), [1, M]);
    affmat          = affparam2mat(tmplH.aff);
    affmat          = [affmat(3:4, :); affmat(1, :); affmat(5:6, :); affmat(2, :)];
    tmplH.project     = [affmat; tmplH.proj; ones(1, M)];
    %% for tracker part
    idxTracker = zeros(N, M);
    tmplTracker = zeros(N, 4, M);
    for i = 1:M
        cumconfT = cumsum(confTracker(:, i));
        idxTracker(:, i) = floor( sum( repmat( rand(1, N), [N, 1] )  > repmat(cumconfT, [1, N]) ) ) + 1;
        tmplTracker(:, :, i) = initTracker(idxTracker(:, i), :, i);
        tmplTracker(:, 4, i) = tmplTracker(:, 4, i) ./ tmplTracker(:, 3, i);
        tmplTracker(:, 3, i) = tmplTracker(:, 3, i) / szH;
        tmplTracker(:, :, i) = tmplTracker(:, :, i) + randn(N, 4).*repmat(varTmpl,[N, 1]);
        tmplTracker(:, 3, i) = tmplTracker(:, 3, i) * szH;
        tmplTracker(:, 4, i) = tmplTracker(:, 4, i) .* tmplTracker(:, 3, i);
        % reserve the best one from the last frame
        [~, j] = max(confTracker(:, i));
        rndIdx = randperm(N);
        tmplTracker(:, :, i) = [initTracker(j, :, i); tmplTracker(rndIdx(1:end-1), :, i)];
    end
    % tmpl 			= initTmpl;
    % N               = size(initTmpl, 1);
    % cumconf         = cumsum(initConf);
    % idx             = floor(sum(repmat(rand(1,N),[N,1]) > repmat(cumconf,[1, N])))+1;
    % tmpl            = initTmpl(idx, :);
    
end



% tmpl(:, 4)      = tmpl(:, 4) ./ tmpl(:, 3);
% tmpl(:, 3)      = tmpl(:, 3) / szH;
% tmpl            = tmpl + randn(N, 4).*repmat(varTmpl,[N, 1]);
% tmpl(:, 3)      = tmpl(:, 3) * szH;
% tmpl(:, 4)      = tmpl(:, 3) .* tmpl(:, 4);

% rndIdx  = randperm(size(tmpl, 1));
% tmpl    = tmpl(rndIdx(1:end - 1), :);
% tmpl    = [initTmpl(i, :); tmpl]; % reserve the best one from the last frame

% idx = (tmpl(:, 3) > 3 & tmpl(:, 4) > 3);
% tmpl = tmpl(idx, :);



