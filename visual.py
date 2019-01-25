import numpy as np
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from qimage2ndarray import gray2qimage, array2qimage
import model
from model import SET_HYPERPARAMETER

interval = 10
latentSpace = 8
restore="DIFF_23jan_ls8_e"
codeNpz="./npz/codes_ls8.npz"

class Render(QLabel):
	def __init__(self, widget):
		QLabel.__init__(self, widget)
		self.setAlignment(Qt.AlignCenter)
		widget.layout.addWidget(self)

	def update(self, data):
		image = QPixmap(array2qimage(data, normalize=True))
		image = image.scaled(400, 300)

		self.setPixmap(image)
		self.setPixmap(QPixmap(image))


class Slider(QSlider):
	def __init__(self, widget, idx, maxv, minv):
		QSlider.__init__(self, Qt.Horizontal)

		self.layout = QHBoxLayout()
		self.label = QLabel()
		self.idx = idx
		self.widget = widget
		self.setMaximum(maxv*interval)
		self.setMinimum(minv*interval)
		self.setValue(0)
		self.setTickPosition(QSlider.TicksBelow)
		self.setTickInterval(5)

		self.label.setText(str(self.value()))
		self.layout.addWidget(self.label)
		self.layout.addWidget(self)
		widget.layout.addLayout(self.layout)
		self.valueChanged.connect(self.valuechange)

	def valuechange(self):
		self.widget.update()
		self.label.setText(str(self.value()/10))

	def getValue(self):
		return self.value()/interval

class Window(QWidget):
	def __init__(self):
		QWidget.__init__(self)
		self.layout = QVBoxLayout()
		SET_HYPERPARAMETER("diffLatentSpace", latentSpace)
		self.data = np.load("./npz/diffs.npz")["arr_1"]
		self.codes = np.load(codeNpz)["arr_1"]
		self.maxs = np.max(self.codes, axis=0)
		self.mins = np.min(self.codes, axis=0)
		self.model = model.emptyModel(restore + "_visual", inputsShape=list(self.data.shape[1:]), use="diff", log=False)
		self.model.restore(restore)
		self.setLayout(self.layout)
		self.setWindowTitle("JTEKT encoder visualization")
		self.sliders = []
		for idx in range(latentSpace):
			self.sliders.append(Slider(self, idx, self.maxs[idx], self.mins[idx]))
		self.render = Render(self)
		self.update()

	def getCode(self):
		code = []
		for slider in self.sliders:
			code.append(slider.getValue())
		return [code]

	def update(self):
		code = self.getCode()
		data = self.model.generate(code, self.data[0:1])[0]
		self.render.update(data)

def main():
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()
