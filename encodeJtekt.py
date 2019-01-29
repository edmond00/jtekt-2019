import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("learningRate", 0.001)
SET_HYPERPARAMETER("latentSpace", 3)
goods = np.load("./npz/dataWithNames.npz")["arr_0"]
bads = np.load("./npz/dataWithNames.npz")["arr_1"]
data = np.concatenate([bads, goods])
model = model.emptyModel("28jan-ls3", inputsShape=list(data.shape[1:]), use="jtekt")

model.restore("16jan-ls3")
model.train(epoch=10, dataset=data)
model.save()
