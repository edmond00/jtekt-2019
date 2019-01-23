import numpy as np
import matplotlib.pyplot as plt

bads = np.load("npz/codes_ls8.npz")["arr_1"]
bads = [bads[:,idx] for idx in range(bads.shape[1])]
goods = np.load("npz/codes_ls8.npz")["arr_0"]
goods = [goods[:,idx] for idx in range(goods.shape[1])]
for idx in range(len(bads)):
	ax = plt.subplot(len(bads),1,idx+1)
	ax.hist(bads[idx], bins=200, color="r", alpha=0.5)
	ax.hist(goods[idx], bins=200, color="b", alpha=0.5)

plt.show()
