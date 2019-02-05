import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 300.0)
SET_HYPERPARAMETER("learningRate", 0.0005)
SET_HYPERPARAMETER("diffLatentSpace", 5)
SET_HYPERPARAMETER("normalize", "individual")

files = np.load("./npz/diffsWithNames.npz")
goods = files["arr_0"]
bads = files["arr_1"]
data = np.concatenate([bads, goods])
model = model.emptyModel("5feb-ls5-inorm_f", inputsShape=list(data.shape[1:]), use="diff")

model.restore("5feb-ls5-inorm_e")
model.train(epoch=100, dataset=data)
model.save()
