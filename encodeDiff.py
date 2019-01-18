import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 50.0)
SET_HYPERPARAMETER("learningRate", 0.005)
#SET_HYPERPARAMETER("diffLatentSpace", 6)
SET_HYPERPARAMETER("diffLatentSpace", 5)
data = np.load("./npz/diffs.npz")["arr_1"]
model = model.emptyModel("DIFF_18jan_ls5_b", inputsShape=list(data.shape[1:]), use="diff")

model.restore("DIFF_18jan_ls5")
model.train(epoch=30, dataset=data)
model.save()
