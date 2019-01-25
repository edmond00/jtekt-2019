import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import *
import cv2
import os

rawNpz = "./npz/dataWithNames.npz"
npz = "./npz/smoothDiffsWithNames.npz"
rows = 20
cols = 8
sampleSize=2000
sampleSize2=8000
nclusters=10
nclusters2=20
plotHistogram=False
plotImg=True

def encode(extractor, clusterer, data):
	kp,descriptor = extractor.detectAndCompute(data, None)
	hist = np.zeros([nclusters], dtype=np.int32)
	if kp is not None:
		for k in kp:
			cluster = clusterer.predict(np.array([[k.angle, k.size]]))[0]
			hist[cluster] += 1
	return hist

raw = np.load(rawNpz)
goodRaws = raw["arr_0"]
badRaws = raw["arr_1"]
allRaws = np.concatenate([goodRaws, badRaws])

allRaws = allRaws.reshape([allRaws.shape[0], allRaws.shape[1], allRaws.shape[2]])
allRaws = np.array(allRaws * 255, dtype = np.uint8)

arr = np.load(npz)
goods = arr["arr_0"]
bads = arr["arr_1"]
alls = np.concatenate([goods, bads])

alls = alls.reshape([alls.shape[0], alls.shape[1], alls.shape[2]])
alls[alls < 0.0] = 0.0
alls[alls > 1.0] = 1.0
alls = np.array(alls * 255, dtype = np.uint8)

extractor = cv2.xfeatures2d.SIFT_create(
	nfeatures=0,
	contrastThreshold=0.06,
	edgeThreshold=45.,
	sigma=1.2
)

sample = np.random.choice(len(alls), sampleSize, replace=False)
descriptors = []
print("prepare sample descriptor")
for idx in tqdm(sample):
	img = alls[idx]
	kp,descriptor = extractor.detectAndCompute(img, None)
	if descriptor is not None:
		#descriptors.append(descriptor)
		for k in kp:
			descriptors.append(np.array([[k.angle, k.size]]))
descriptors = np.concatenate(descriptors)
print(descriptors.shape)

clusterer = KMeans(n_clusters=nclusters)
clusterer.fit(descriptors)

sample2 = np.random.choice(len(alls), sampleSize2, replace=False)
codes = []
print("encode data")
for idx in tqdm(sample2):
	codes.append(encode(extractor, clusterer, alls[idx]))
clusterer2 = KMeans(n_clusters=nclusters2)
clusterer2.fit(codes)
labels = clusterer2.labels_
idxByLabels = [sample2[labels == i] for i in range(0, labels.max()+1)]

plt.axis("off")
if plotHistogram is True:
	n = 1
	for y in range(rows):
		print("cluster %d : %d" % (y,len(idxByLabels[y])))
		for x in range(cols):
			if y < len(idxByLabels) and x < len(idxByLabels[y]):
				idx = idxByLabels[y][x]
				img = alls[idx]
				kp,descriptor = extractor.detectAndCompute(img, None)
				hist = np.zeros([nclusters], dtype=np.int32)
				#if descriptor is not None:
				#	for description in descriptor:
				#		cluster = clusterer.predict([description])[0]
				#		hist[cluster] += 1
				if kp is not None:
					for k in kp:
						cluster = clusterer.predict(np.array([[k.angle, k.size]]))[0]
						hist[cluster] += 1
				img = cv2.drawKeypoints(img, kp, img,
					flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
				ax = plt.subplot(rows, cols*2, n)
				plt.imshow(img, "gray")
				ax = plt.subplot(rows, cols*2, n+1)
				plt.bar(list(range(nclusters)), hist)
				n += 2
	plt.show()

if plotImg is True:
	n = 1
	for y in range(rows):
		print("cluster %d : %d" % (y,len(idxByLabels[y])))
		for x in range(cols):
			if y < len(idxByLabels) and x < len(idxByLabels[y]):
				idx = idxByLabels[y][x]
				img = alls[idx]
				ax = plt.subplot(rows, cols*2, n)
				plt.imshow(allRaws[idx], "gray")
				ax = plt.subplot(rows, cols*2, n+1)
				plt.imshow(img, "gray")
				n += 2
	plt.show()
