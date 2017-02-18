function shouldUpdate = UpdateDifferenceJudger(model, opt)
if (model.lastProb - model.secondProb < opt.UpdateDifferenceJudger.thresold)
    shouldUpdate = true;
    fprintf('%f\t%f\n', model.lastProb, model.secondProb);
else
    shouldUpdate = false;
end
