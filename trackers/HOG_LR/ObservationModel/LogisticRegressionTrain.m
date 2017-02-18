function model = LogisticRegressionTrain(dataPos, dataNeg, opt, model)

featPos         = dataPos.feat;
featNeg         = dataNeg.feat;
numPos          = size(featPos, 2);
numNeg          = size(featNeg, 2);
featDim         = size(featPos, 1);
num             = numPos + numNeg;

feat            = [featPos, featNeg];
data.feat       = feat;

feat            = [feat; ones(1, num)];     % add bias term

label           = zeros(1, num);
label(1:numPos) = 1;

if nargin <= 3
    model.w     = randn(featDim + 1, 1) / (featDim + 1);
    maxIter         = 1000;
else
    maxIter = 20;
end

lambda          = 1e-2;
iter            = 0; 
lr              = 1 / num;
alpha           = 0.99; % Parameter for momentum
deltaW          = 0;   
while (iter < maxIter)
    iter        = iter + 1;
    
    loss        = label - LogisticRegressionTest(data, model);
    deltaW       = lr * (feat * loss' - lambda * model.w) + alpha * deltaW;
%     deltaW       = lr * (feat * loss' - lambda * model.w);
    model.w     = model.w + deltaW;
%     disp(norm(loss));
end

end