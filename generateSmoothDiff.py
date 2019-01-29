import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 50.0)
SET_HYPERPARAMETER("diffLatentSpace", 30)
data = np.load("./npz/diffsWithNames.npz")
goods = data["arr_0"]
bads = data["arr_1"]
model = model.emptyModel("generateSmoothDiff", inputsShape=list(goods[0].shape), use="diff", log=False)

model.restore("28jan_ls30")
trainDiffs = model.reproduce(goods)
testDiffs = model.reproduce(bads)
np.savez("./npz/smoothDiffsWithNames_ls30.npz", trainDiffs, testDiffs, data["arr_2"], data["arr_3"])

