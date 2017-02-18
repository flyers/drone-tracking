function prob = SOSVMTest(data, ~)

data.tmpl(:,1)  = data.tmpl(:,1) - data.tmpl(:, 3) / 2;
data.tmpl(:,2)  = data.tmpl(:,2) - data.tmpl(:, 4) / 2;

prob = mexSOSVMLearn(data.tmpl, data.feat', 'test')';