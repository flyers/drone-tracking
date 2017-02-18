function model = SOSVMTrain(dataPos, dataNeg, opt, model)

featPos         = dataPos.feat;
featNeg         = dataNeg.feat;
rectPos         = dataPos.tmpl;
rectNeg         = dataNeg.tmpl;

data.feat       = [featPos, featNeg];
% data.feat = bsxfun(@rdivide, data.feat, sqrt(sum(data.feat.*data.feat)));
data.feat = data.feat';

data.tmpl       = [rectPos; rectNeg];
data.tmpl(:,1)  = data.tmpl(:,1) - data.tmpl(:, 3) / 2;
data.tmpl(:,2)  = data.tmpl(:,2) - data.tmpl(:, 4) / 2;

if (opt.SOSVM.kernel == 1)
    conf = '-t 1 -g -0.1';
elseif (opt.SOSVM.kernel == 0)
    conf = '-t 0 ';
else
    conf = '';
end
if nargin <= 3
    mexSOSVMLearn(data.tmpl, data.feat, 'batchTrain', [conf, '-b 100 -c 100.0']);
    model = [];
else
    mexSOSVMLearn(data.tmpl, data.feat, 'onlineTrain');
end
