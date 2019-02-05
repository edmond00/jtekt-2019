import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("learningRate", 0.001)
SET_HYPERPARAMETER("latentSpace", 1)
goods = np.load("./npz/dataWithNames.npz")["arr_0"]
bads = np.load("./npz/dataWithNames.npz")["arr_1"]
data = np.concatenate([bads, goods])
model = model.emptyModel("4feb-ls1", inputsShape=list(data.shape[1:]), use="jtekt")

#model.restore("16jan-ls3")
model.train(epoch=20, dataset=data)
model.save()
