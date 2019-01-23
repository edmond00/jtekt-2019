import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 50.0)
SET_HYPERPARAMETER("diffLatentSpace", 8)
model = model.npzModel("generateEncode", "./npz/diffs.npz", use="diff", log=False)

model.restore("DIFF_23jan_ls8_c")
testEncoded = model.encode(model.testSet)
trainEncoded = model.encode(model.trainSet)
np.savez("./npz/codes_ls8.npz", trainEncoded, testEncoded)

