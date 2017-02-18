function shouldUpdate = UpdateCorrectJudger(model, opt)
%model.lastProb
if (model.lastProb > opt.ClassificationScoreJudger.thresold)
    shouldUpdate = true;
else
    shouldUpdate = false;
end
