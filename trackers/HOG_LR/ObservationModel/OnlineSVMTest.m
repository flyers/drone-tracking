function prob = OnlineSVMTest(data, model)

prob = -(model.w * data.feat + model.Bias);