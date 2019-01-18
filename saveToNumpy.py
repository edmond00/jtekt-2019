from data import Data

data = Data.dataFromFiles()
data.saveToNumpy("./npz/dataWithNames.npz")
