import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 300.0)
SET_HYPERPARAMETER("diffLatentSpace", 6)
data = np.load("./npz/diffsWithNames.npz")
goods = data["arr_0"]
bads = data["arr_1"]
model = model.emptyModel("generateEncode", use="diff", log=False, inputsShape=list(goods[0].shape))

model.restore("29jan_vae_ls6_d")
testEncoded = model.encode(bads)
trainEncoded = model.encode(goods)
np.savez("./npz/codesWithNames_vae_ls6.npz", trainEncoded, testEncoded, data["arr_2"], data["arr_3"])

