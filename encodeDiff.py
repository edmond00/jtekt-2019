import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 300.0)
SET_HYPERPARAMETER("learningRate", 0.0003)
SET_HYPERPARAMETER("diffLatentSpace", 6)
data = np.load("./npz/diffs.npz")["arr_1"]
model = model.emptyModel("29jan_vae_ls6_e", inputsShape=list(data.shape[1:]), use="diff")

model.restore("29jan_vae_ls6_d")
model.train(epoch=50, dataset=data)
model.save()
