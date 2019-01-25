import numpy as np
from model import emptyModel,SET_HYPERPARAMETER
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.cluster import *
pylab.rcParams['figure.figsize'] = 20,20

data = np.load("npz/dataWithNames.npz")
goods = data["arr_0"]
bads = data["arr_1"]
allData = np.concatenate([goods,bads])
goodNames = data["arr_2"]
badNames = data["arr_3"]
allNames = np.concatenate([goodNames,badNames])

SET_HYPERPARAMETER("contrast", 50.)
SET_HYPERPARAMETER("diffLatentSpace", 8)
SET_HYPERPARAMETER("latentSpace", 3)
rawEncoder = emptyModel("raw_encoder", use="jtekt", inputsShape=list(allData.shape[1:]), log=False)
diffEncoder = emptyModel("diff_encoder", use="diff", inputsShape=[76,134,1], log=False)
print("restore diff encoder")
diffEncoder.restore("DIFF_23jan_ls8_e")
print("restore raw encoder")
rawEncoder.restore("16jan-ls3")

def encodeAndCluster(plotResult=False, plotOne=False, csv=None):
    dataset = allData
    imgs = dataset.reshape(dataset.shape[0:3])
    print("get reproductions...")
    reproductions = rawEncoder.reproduce(dataset)
    reproductionImgs = reproductions.reshape(reproductions.shape[0:3])
    print("get diffs...")
    diffs = rawEncoder.getDiff(dataset)
    diffImgs = diffs.reshape(diffs.shape[0:3])
    print("get encoding...")
    encodedDiffs = diffEncoder.encode(diffs)
    print("get reproduction...")
    diffReproductions = diffEncoder.reproduce(diffs)
    diffReproductionImgs = diffReproductions.reshape(diffReproductions.shape[0:3])

    print("cluster...")
    ridx = np.random.choice(len(encodedDiffs), 1000, replace=False)
    #encodedDiffs = encodedDiffs[ridx]
    #clustering = MeanShift(bandwidth=10).fit(encodedDiffs)
    clustering = Ward(n_clusters=50).fit(encodedDiffs)
    labels = clustering.labels_

    nclusters = labels.max()
    print("\nNumber clusters : %d" % nclusters)

    allClusters = [np.arange(len(labels))[labels == i] for i in range(0, labels.max()+1)]
    clusterMinSize = 1
    nBigClusters = 0
    unclassifyData = 0

    clusters = []
    bigLabels = []
    idx = 0
    for cluster in allClusters:
        np.random.shuffle(cluster)
        if len(cluster) < clusterMinSize:
            unclassifyData += len(cluster)
            bigLabels.append(-1)
        else:
            nBigClusters += 1
            clusters.append(cluster)
            bigLabels.append(idx)
            idx += 1

    sizes = [len(cluster) for cluster in clusters]
    print("\nNumber big clusters : %d" % nBigClusters)
    print("small cluster data : %d" % unclassifyData)
    print("biggest clusters : %d" % max(sizes))
    print("second biggest clusters : %d" % sorted(sizes)[-2])
    print("third biggest clusters : %d" % sorted(sizes)[-3])
    print("mean clusters size : %d" % np.mean([len(cluster) for cluster in clusters]))
    print("median clusters size : %d\n" % np.median([len(cluster) for cluster in clusters]))


    if plotResult:
        np.random.shuffle(clusters)
        n = 1
#        plt.figure(figsize=(nclusters,20))
        plt.suptitle("JTEKT unsupervised clustering")
        rows = min(15, nclusters)
        cols = 6
        for i in range(rows):
            print("Cluster %d size : %d" % (i, len(clusters[i])))
            for j in range(cols):
                ax = plt.subplot(rows, cols*2, n)
                ax.axis('off')
                if j < len(clusters[i]):
                    img = imgs[clusters[i][j]]
                    ax.imshow(img, cmap="gray", vmin=0, vmax=imgs.max())
                n += 1
                ax = plt.subplot(rows, cols*2, n)
                ax.axis('off')
                if j < len(clusters[i]):
                    img = diffReproductionImgs[clusters[i][j]]
                    ax.imshow(img, cmap="gray", vmax=diffReproductionImgs.max())
                n += 1
#        plt.tight_layout()
        plt.show()

    if plotOne:
        for column in range(4):
            idx = np.random.randint(imgs.shape[0])
            ax = plt.subplot(4, 4, column*4+1)
            ax.axis('off')
            img = imgs[idx]
            ax.imshow(img, cmap="gray", vmin=0, vmax=img.max())

            ax = plt.subplot(4, 4, column*4+2)
            ax.axis('off')
            img = reproductionImgs[idx]
            ax.imshow(img, cmap="gray", vmin=0, vmax=img.max())

            ax = plt.subplot(4, 4, column*4+3)
            ax.axis('off')
            img = diffImgs[idx]
            ax.imshow(img, cmap="gray", vmin=0, vmax=img.max())

            ax = plt.subplot(4, 4, column*4+4)
            ax.axis('off')
            img = diffReproductionImgs[idx]
            ax.imshow(img, cmap="gray", vmin=0, vmax=img.max())
        plt.show()

    if csv is not None:
        file = open(csv,"w") 
        file.write("file,cluster\n")
        for idx in range(len(allNames)):
            file.write("%s,%d\n" % (allNames[idx],bigLabels[labels[idx]]))
        file.close()


if __name__ == "__main__":
    encodeAndCluster(plotResult=True, plotOne=True, csv=None)
