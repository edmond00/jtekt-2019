import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("contrast", 50.0)
SET_HYPERPARAMETER("diffLatentSpace", 5)
model = model.npzModel("generateEncode", "./npz/diffs.npz", use="diff", log=False)

model.restore("DIFF_18jan_ls5_b")
testEncoded = model.encode(model.testSet)
trainEncoded = model.encode(model.trainSet)
np.savez("./npz/codes_ls5.npz", trainEncoded, testEncoded)

