import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 50.0)
SET_HYPERPARAMETER("learningRate", 0.001)
SET_HYPERPARAMETER("diffLatentSpace", 3)
data = np.load("./npz/diffs.npz")["arr_1"]
model = model.emptyModel("24jan_ls3", inputsShape=list(data.shape[1:]), use="diff")

#model.restore("")
model.train(epoch=200, dataset=data)
model.save()
