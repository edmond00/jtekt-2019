import model
import numpy as np
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("latentSpace", 3)

model = model.npzModel("16jan-ls3-generateDiff", "./npz/dataWithNames.npz", log=False, use="jtekt")

model.restore("16jan-ls3")
trainDiffs = model.getDiff(model.trainSet)
testDiffs = model.getDiff(model.testSet)
np.savez("./npz/diffs.npz", trainDiffs, testDiffs)

