import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 50.0)
SET_HYPERPARAMETER("diffLatentSpace", 8)
data = np.load("./npz/diffsWithNames.npz")
goods = data["arr_0"]
bads = data["arr_1"]
model = model.emptyModel("generateFeatures", use="diff", log=False, inputsShape=list(goods[0].shape))

model.restore("DIFF_23jan_ls8_e")
testFeatures = model.getFeatures(bads)
trainFeatures = model.getFeatures(goods)
np.savez("./npz/featuresWithNames_ls8.npz", trainFeatures, testFeatures, data["arr_2"], data["arr_3"])
