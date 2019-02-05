from data import Data
import numpy as np
import os
from sklearn.metrics.cluster import adjusted_rand_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def getLabels(data, names, labels):
    trueLabels = []
    predLabels = []
    for raw in data.rawData:
        trueLabel = np.where(raw.labels != 0.)[0][0]
        idx = np.where(names == raw.name)[0][0]
        predLabel = labels[idx]
        trueLabels.append(trueLabel)
        predLabels.append(predLabel)
    return trueLabels,predLabels

npz = os.listdir("npz/results")

data = Data.dataFromFiles()
for filename in npz:
    filename = os.path.join("npz/results", filename)
    results = np.load(filename)
    labels = results["arr_1"]
    names = results["arr_0"]
    trueLabels, predLabels = getLabels(data, names, labels)
    score = adjusted_rand_score(trueLabels, predLabels)
    print("%s : %f" % (filename, score))

