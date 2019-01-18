import numpy as np
import model
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("diffLatentSpace", 6)
data = np.load("./npz/diffs.npz")["arr_1"]
model = model.emptyModel("DIFF_17jan_ls6_g", inputsShape=list(data.shape[1:]), use="diff", log=False)
model.restore("DIFF_17jan_ls6_f")
code = [[0,0,0,0,0,0]]
result = model.generate(code, data[0:1])
print(result)
print(result.shape)

