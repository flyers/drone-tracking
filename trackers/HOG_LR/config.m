% global parameter
globalParam = struct(...
    'MotionModel',                  @ParticleFilterMotionModel, ...
    'FeatureExtractor',             @HogRawPixelNormExtractor, ...
    'ConfidenceJudger',             @ClassificationScoreJudger, ...
    'ObservationModelTrain',        @LogisticRegressionTrain, ...
    'ObservationModelTest',         @LogisticRegressionTest, ...
    'NegSampler',                   @NegSlidingWindowSampler, ...
    'PosSampler',                   @PosSlidingWindowSampler ... 
);



opt.condenssig = 0.05;
opt.useNormalSize = true;
opt.normalWidth = 640;
opt.normalHeight = 360;

opt.FeatureExtractor.tmplsize = [32, 32];
opt.FeatureExtractor.NumBins = 8; 

opt.Sampler.NegSlidingWindowSampler.NegSlidingH = 100;
opt.Sampler.NegSlidingWindowSampler.NegSlidingW = 100;
opt.Sampler.NegSlidingWindowSampler.NegStride = 5;
opt.Sampler.NegSlidingWindowSampler.excludeNegRatio = 0.3;

opt.Sampler.PosSlidingWindowSampler.PosSlidingH = 5;
opt.Sampler.PosSlidingWindowSampler.PosSlidingW = 5;

opt.MotionModel.ParticleFilterMotionModel.N = 400;
opt.MotionModel.ParticleFilterMotionModel.affsig = [12,12, 0.01, 0.005];

opt.ClassificationScoreJudger.thresold = 0.9;

opt.useFirstFrame = true;
% opt.ClassificationScoreJudger.thresold = 10; % SOSVM

opt.visualization = 0;

% for extracting background feature points
opt.feature.normalWidth = 640;
opt.feature.normalHeight = 360;
