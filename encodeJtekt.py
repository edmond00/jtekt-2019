import model
from model import SET_HYPERPARAMETER

SET_HYPERPARAMETER("learningRate", 0.01)
SET_HYPERPARAMETER("latentSpace", 3)
model = model.npzModel("16jan-ls3", "./npz/dataWithNames.npz", use="jtekt")

#model.restore("16jan-ls3")
model.train(epoch=50)
model.save()
