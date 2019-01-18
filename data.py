import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import numpy as np
import csv
import sys
from tqdm import tqdm
from PIL import Image
import os

imageDirectory = "./data/jpg/"
csvList = ["./data/utf8_data_mibunrui.csv", "./data/utf8_data_ng_j.csv", "./data/utf_data_ng.csv", "./data/utf_data_ok.csv"]
labels = ["良品", "横細線", "横太線", "横2本線", "横短線", "全体黒", "横縞線", "縦短線", "縦長線", "縦縞線",
          "斑点", "斜め線", "曲線", "黒が丘", "矩形", "細紋", "波線", "その他", "不明", "画像不良"]
maxData = None

random.seed(0)

def resized(n):
    return int(n/resizeFactor)

class IgnoreList:

    def __init__(self):
        f = open("./data/ignoreList.csv", "r")
        self.names = f.read().splitlines()

    def toIgnore(self, line):
        for name in self.names:
            if name in line:
                return True
        return False

ignoreList = IgnoreList()
        

class Item:
    
    def __init__(self):
        pass

    def readLine(self, line):
        splits = line.split(",")
        self.name = splits[0].split("/")[-1]
        self.path = splits[0]
        jpg = Image.open(imageDirectory + splits[0])
        self.originalImage = jpg
        self.labels = np.array([float(s) for s in splits[1:]], dtype = np.float32)
        self.data = np.array(jpg, dtype=np.float32)
        self.colorMean()
        self.regularize()

    def saveImage(self, directory):
        self.originalImage.save(os.path.join(directory, self.name))

    def normalize(self):
        self.data = (self.data - self.data.mean()) / (self.data.max() - self.data.min())

    def regularize(self):
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

    def crop(self, up, down, left, right):
        self.data = self.data[up:-down,left:-right,:]

    def colorMean(self):
        self.data = (self.data[:,:,0] + self.data[:,:,1] + self.data[:,:,2]) / 3
        self.data = self.data.reshape([self.data.shape[0], self.data.shape[1], 1])

    def hasLabel(self, label):
        idx = labels.index(label)
        return self.labels[idx] == 1

    @classmethod
    def dataFromFile(cls, line, idx):
        image = cls()
        image.readLine(line)
        image.idx = idx
        return image

    @classmethod
    def clone(cls, origin):
        image = cls()
        image.__dict__ = origin.__dict__.copy()
        return image

class Data:

    def __init__(self):
        self.defaultThreshold = 0.15
        pass

    def sortByLabel(self):
        self.dataByLabel = {}
        for label in tqdm(labels):
            tmp = []
            for item in self.rawData:
                if item.hasLabel(label):
                    tmp.append(item)
            self.dataByLabel[label] = tmp

    def splitTrainTest(self, testPercent):
        self.testByLabel = {}
        self.trainByLabel = {}
        self.testAll = []
        self.trainAll = []
        self.goodOnly = []
        self.badOnly = []
        for label in tqdm(labels):
            n = int(len(self.dataByLabel[label]) *testPercent / 100)
            if label == "良品":
                self.goodOnly += [item.idx for item in self.dataByLabel[label]]
            else:
                self.badOnly += [item.idx for item in self.dataByLabel[label]]
            self.testByLabel[label] = [item.idx for item in random.sample(self.dataByLabel[label], n)]
            self.trainByLabel[label] = [item.idx for item in self.dataByLabel[label] if item.idx not in self.testByLabel[label]]
            self.testAll = self.testAll + self.testByLabel[label]
            self.trainAll = self.trainAll + self.trainByLabel[label]

    def getAll(self, dataset):
        inputs = []
        outputs = []
        for idx in dataset:
            inputs.append(self.rawData[idx].data)
            outputs.append(self.rawData[idx].labels.reshape([-1]))
        return np.array(inputs), np.array(outputs)
            

    def batch(self, dataset, size):
        inputs = []
        outputs = []
        for i in range(size):
            label = labels[i%len(labels)]
            idx = random.choice(dataset[label])
            inputs.append(self.rawData[idx].data)
            outputs.append(self.rawData[idx].labels.reshape([-1]))
        return np.array(inputs), np.array(outputs)

    def unbalancedBatch(self, dataset, size):
        inputs = []
        outputs = []
        idxs = random.choice(dataset[label], size)
        for idx in idxs:
            inputs.append(self.rawData[idx].data)
            outputs.append(self.rawData[idx].labels.reshape([-1]))
        return np.array(inputs), np.array(outputs)

    def train(self, model, epoch, batchSize, learningRate, dropout=0.0, regularization=0.0):
        printBy = 10
        print("Trainingg...\n")
        for i in range(epoch):
            bi, bo = self.batch(self.trainByLabel, batchSize)
            trainCost = model.train(bi, bo, batchSize, learningRate, dropout, regularization)
            if i % printBy == 0:
                tbi, tbo = self.batch(self.testByLabel, batchSize)
                testCost = model.getCost(tbi, tbo, batchSize, test=True)
                mAP = self.map(model, batchSize, self.trainByLabel)
                testMAP = self.map(model, batchSize, self.testByLabel)
                model.writeMap(mAP)
                model.writeTestMap(testMAP)
                print("COST: train = " + str(trainCost) + ", test = " + str(testCost), end="\r")


    def cost(self, model, batchSize, dataset = None):
        if dataset is None:
            dataset = self.testByLabel
        bi, bo = self.batch(dataset, batchSize)
        cost = model.getCost(bi, bo, batchSize)
        return cost

    def test(self, model, batchSize, threshold=None, dataset=None):
        if dataset is None:
            dataset = self.testByLabel
        bi, bo = self.batch(dataset, batchSize)
        predictions = model.predict(bi, batchSize)
        TP = np.zeros(len(bo[0]))
        FP = np.zeros(len(bo[0]))
        TN = np.zeros(len(bo[0]))
        FN = np.zeros(len(bo[0]))
        for i in range(len(bo)):
            for j in range(len(bo[i])):
                if threshold is None:
                    threshold = self.defaultThreshold
                if predictions[i][j] > threshold: #POSITIVE
                    if bo[i][j] == 1.: #TRUE
                        TP[j] += 1
                    else: #FALSE
                        FP[j] += 1
                else: #NEGATIVE
                    if bo[i][j] == 1.: #FALSE
                        FN[j] += 1
                    else: #TRUE
                        TN[j] += 1
        return TP,FP,TN,FN

    def metrics(self, model, batchSize, threshold=None, dataset=None):
        TP, FP, TN, FN = self.test(model, batchSize, threshold, dataset)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        np.nan_to_num(precision, False)
        np.nan_to_num(recall, False)
        np.nan_to_num(f1, False)
        return precision, recall, f1

    def averagePrecision(self, model, batchSize, dataset=None):
        ap = np.zeros(len(labels))
        lastRecall = np.zeros(len(labels))
        thresholds = np.flip(np.arange(0,1.1,0.1), axis = 0)
        for t in thresholds:
            precision, recall, f1 = self.metrics(model, batchSize, t, dataset)
            ap += (precision * (recall - lastRecall))
            lastRecall = recall
        result = {}
        for i in range(len(labels)):
            result[labels[i]] = ap[i]
        return result

    def map(self, model, batchSize, dataset=None):
        ap = self.averagePrecision(model, batchSize, dataset)
        ap = [v for k,v in ap.items()]
        return float(sum(ap) / len(ap))

    def saveImages(self, goodDir, badDir):
        for item in tqdm(self.rawData):
            if item.hasLabel("良品"):
                item.saveImage(goodDir)
            else:
                item.saveImage(badDir)

    def saveToNumpy(self, filename):
        goods = []
        bads = []
        goodNames = []
        badNames = []
        print("Save to numpy...")
        for item in tqdm(self.rawData):
            if item.hasLabel("良品"):
                goods.append(item.data)
                goodNames.append(item.path)
            else:
                bads.append(item.data)
                badNames.append(item.path)
        goods = np.array(goods)
        goodNames = np.array(goodNames)
        bads = np.array(bads)
        badNames = np.array(badNames)
        np.savez(filename, goods, bads, goodNames, badNames)


    @classmethod
    def dataFromFiles(cls):
        result =  cls()
        result.rawData =cls.getData(csvList) 
        print("dispatch by label...")
        result.sortByLabel()
        print("split train/test...")
        result.splitTrainTest(20)
        result.inputLen = result.rawData[0].data.size
        result.outputLen = len(labels)
        return result

    @classmethod
    def clone(cls, origin):
        rawData = origin.rawData
        for i in range(len(origin.rawData)):
            rawData[i] = Item.clone(origin.rawData[i])
        result = cls()
        result.__dict__ = origin.__dict__.copy()
        result.rawData = rawData
        return result

    @classmethod
    def getData(cls, filenames):
        rawData = []
        for filename in filenames:
            print("load data from " + filename + " ...")
            numberLines = sum(1 for line in open(filename, "r"))
            f = open(filename, "r")
            line = f.readline()
            for nu in tqdm(range(numberLines)):
                if (nu > 0
                    and (maxData is None or len(rawData) < maxData)
                    and ignoreList.toIgnore(line) == False):
                    rawData.append(Item.dataFromFile(line, len(rawData)))
                line = f.readline()
        return rawData
