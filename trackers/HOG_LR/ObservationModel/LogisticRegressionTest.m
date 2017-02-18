function prob = LogisticRegressionTest(data, model)

prob = 1 ./ (1 + exp(- (model.w(1:end-1)' * data.feat) - model.w(end)));
% prob = 1 ./ (1 + exp(- (model.w' * data.feat)));