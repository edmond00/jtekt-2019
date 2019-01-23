import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 50.0)
SET_HYPERPARAMETER("learningRate", 0.0005)
SET_HYPERPARAMETER("diffLatentSpace", 8)
data = np.load("./npz/diffs.npz")["arr_1"]
model = model.emptyModel("DIFF_23jan_ls8_d", inputsShape=list(data.shape[1:]), use="diff")

model.restore("DIFF_23jan_ls8_c")
model.train(epoch=200, dataset=data)
model.save()
