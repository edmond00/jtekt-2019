import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("latentSpace", 1)

data = np.load("./npz/dataWithNames.npz")
goods = data["arr_0"]
bads = data["arr_1"]
model = model.emptyModel("generateDiff", inputsShape=list(goods[0].shape), log=False, use="jtekt")

model.restore("4feb-ls1")
trainDiffs = model.getDiff(goods)
testDiffs = model.getDiff(bads)
np.savez("./npz/diffsWithNames.npz", trainDiffs, testDiffs, data["arr_2"], data["arr_3"])

