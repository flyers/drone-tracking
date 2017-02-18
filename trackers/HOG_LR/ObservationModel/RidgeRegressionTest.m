function prob = RidgeRegressionTest(data, model)

prob = model.beta' * data.feat;
% prob = data.feat' * model.beta;