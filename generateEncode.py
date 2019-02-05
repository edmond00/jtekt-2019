import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 300.0)
SET_HYPERPARAMETER("diffLatentSpace", 12)
SET_HYPERPARAMETER("normalize", "individual")

data = np.load("./npz/diffsWithNames.npz")
goods = data["arr_0"]
bads = data["arr_1"]
model = model.emptyModel("generateEncode", use="diff", log=False, inputsShape=list(goods[0].shape))

model.restore("5feb-ls12-inorm_d")
testEncoded = model.encode(bads)
trainEncoded = model.encode(goods)
np.savez("./npz/codesWithNames_inorm.npz", trainEncoded, testEncoded, data["arr_2"], data["arr_3"])

