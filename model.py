import os
import numpy as np
import tensorflow as tf
import types
from tqdm import tqdm
from layers  import *

CONFIG = types.SimpleNamespace()
CONFIG.saveDir = "./saves/"
CONFIG.diffSaveDir = "./diffSaves/"
CONFIG.logDir = "./logs/"
CONFIG.maxImages = 200

HYPERPARAMETER = types.SimpleNamespace()
HYPERPARAMETER.learningRate = 0.001
HYPERPARAMETER.batchSize = CONFIG.maxImages
HYPERPARAMETER.epoch = 500
HYPERPARAMETER.latentSpace = 15
HYPERPARAMETER.diffLatentSpace = 5
HYPERPARAMETER.contrast = 100

def SET_HYPERPARAMETER(key, value):
    global HYPERPARAMETER
    HYPERPARAMETER.__dict__[key] = value

def jtektModel(name, data, log=True, shuffle=True):
    rawTrainSet,_ = data.getAll(data.goodOnly)
    rawTestSet,_ = data.getAll(data.badOnly)
    if shuffle is True:
        np.random.shuffle(rawTrainSet)
        np.random.shuffle(rawTestSet)
    trainSet = rawTrainSet
    testSet = rawTestSet
    return Model(name, trainSet, testSet, log=log)

def mnistModel(name, log=True):
    (rawTrainSet, trainLabels),(rawTestSet, testLabels) = tf.keras.datasets.mnist.load_data()
    trainSet = np.expand_dims(rawTrainSet, 3) / 255
    testSet = np.expand_dims(rawTestSet, 3) / 255
    return Model(name, trainSet, testSet, log=log)

def npzModel(name, path, use, log=True, shuffle=True):
    data = np.load(path)
    trainSet = data['arr_0']
    testSet = data['arr_1']
    if shuffle is True:
        np.random.shuffle(trainSet)
        np.random.shuffle(testSet)
    return Model(name, trainSet, testSet, use, log=log)

def emptyModel(name, use, inputsShape, log=True):
    return Model(name, None, None, use, log=log, inputsShape=inputsShape)

class Model:
    
    def jtektEncoder(self):
        self.use = "jtekt"
        #ENCODE
        InputLayer(
            self,
            self.placeholder.inputs,
            self.inputsShape,
            #crop=[5,15,10,15],
            resize=60)
        ConvutionalLayer(self, [2,2,1,32], [1,2,2,1], activation=tf.nn.leaky_relu)
        ConvutionalLayer(self, [2,2,32,64], [1,2,2,1], activation=tf.nn.leaky_relu)
        Reshape(self, [self.batchSize, self.layers[-1].len()])
        Dense(self, HYPERPARAMETER.latentSpace, activation=tf.nn.leaky_relu)

        self.encoder = self.outputs()

        #DECODE
        Dense(self, self.layers[-2].len(), activation=tf.nn.leaky_relu)
        Reshape(self, [self.batchSize] + self.layers[-4].shape)
        DeconvutionalLayer(self, [2,2,32,64], [self.batchSize] + self.layers[1].shape, [1,2,2,1], activation=tf.nn.leaky_relu)
        DeconvutionalLayer(self, [2,2,1,32], [self.batchSize] + self.inputsShape, [1,2,2,1], activation=tf.nn.sigmoid)

    def diffEncoder(self):
        self.use = "diff"
        #ENCODE
        InputLayer(
            self,
            self.placeholder.inputs,
            self.inputsShape,
            contrast=HYPERPARAMETER.contrast,
            normalize=True)
        ConvutionalLayer(self, [2,2,1,32], [1,2,2,1], activation=tf.nn.leaky_relu)
        ConvutionalLayer(self, [2,2,32,64], [1,2,2,1], activation=tf.nn.leaky_relu)
        Reshape(self, [self.batchSize, self.layers[-1].len()])
        Dense(self, HYPERPARAMETER.diffLatentSpace, activation=tf.nn.leaky_relu)

        self.encoder = self.outputs()
        self.layers[-1].outputs2 = self.placeholder.code

        #DECODE
        Dense(self, self.layers[-2].len(), activation=tf.nn.leaky_relu)
        Reshape(self, [self.batchSize] + self.layers[-4].shape)
        DeconvutionalLayer(self, [2,2,32,64], [self.batchSize] + self.layers[1].shape, [1,2,2,1], activation=tf.nn.leaky_relu)
        DeconvutionalLayer(self, [2,2,1,32], [self.batchSize] + self.inputsShape, [1,2,2,1], activation=tf.nn.tanh)

#    def diffEncoder(self):
#        self.use = "diff"
#        #ENCODE
#        InputLayer(
#            self,
#            self.placeholder.inputs,
#            self.inputsShape,
#            contrast=HYPERPARAMETER.contrast,
#            normalize=True)
#        ConvutionalLayer(self, [2,2,1,32], [1,2,2,1], activation=tf.nn.leaky_relu)
#        ConvutionalLayer(self, [2,2,32,64], [1,2,2,1])
#        Reshape(self, [self.batchSize, self.layers[-1].len()])
#        Dense(self, HYPERPARAMETER.diffLatentSpace, activation=tf.nn.sigmoid)
#
#        self.encoder = self.outputs()
#        self.layers[-1].outputs2 = self.placeholder.code
#
#        #DECODE
#        Dense(self, self.layers[-2].len(), activation=tf.nn.leaky_relu)
#        Reshape(self, [self.batchSize] + self.layers[-4].shape)
#        DeconvutionalLayer(self, [2,2,32,64], [self.batchSize] + self.layers[1].shape, [1,2,2,1], activation=tf.nn.leaky_relu)
#        DeconvutionalLayer(self, [2,2,1,32], [self.batchSize] + self.inputsShape, [1,2,2,1], activation=lambda x:tf.nn.relu(tf.nn.tanh(x)))

    def __init__(self, name, trainSet, testSet, use="jtekt", log=True, inputsShape=None):
        tf.reset_default_graph()
        self.name = name
        self.layers = []
        self.weights = []
        self.bias = []
        self.layersByType = {}

        self.log = log
        self.trainSet = trainSet
        self.testSet = testSet
        if inputsShape is not None:
            self.inputsShape = inputsShape
        else:
            self.inputsShape = list(self.trainSet[0].shape)
        print("inputs shape : " + str(self.inputsShape))

        self.initInitialization()
        self.initPlaceholder()

        if use == "diff":
            self.diffEncoder()
        else:
            self.jtektEncoder()

        #DIFF
        self.diff = tf.abs(self.inputs() - self.outputs())

        self.cost = tf.reduce_sum(tf.square(self.inputs() - self.outputs()))
        self.optim = tf.train.AdamOptimizer(self.placeholder.learningRate).minimize(self.cost)

        if self.log:
            self.initSummaries()

        self.session = tf.Session()

        if self.log:
            self.writer = tf.summary.FileWriter(
                os.path.join(CONFIG.logDir, self.name),
                self.session.graph)
        self.saver = tf.train.Saver()
        self.reinit()

    def prefix(self):
        if self.use == "diff":
            return "DIFF/"
        else:
            return ""

    def reinit(self):
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

    def save(self):
        if self.use == "diff":
            self.saver.save(self.session, os.path.join(CONFIG.diffSaveDir, self.name))
        else:
            self.saver.save(self.session, os.path.join(CONFIG.saveDir, self.name))

    def restore(self, path):
        if self.use == "diff":
            self.saver.restore(self.session, os.path.join(CONFIG.diffSaveDir, path))
        else:
            self.saver.restore(self.session, os.path.join(CONFIG.saveDir, path))

    def initInitialization(self):
        self.winit = tf.glorot_uniform_initializer()
#        self.winit = tf.truncated_normal_initializer(stddev=0.03)
        self.convinit = tf.truncated_normal_initializer(stddev=0.03)
#        self.convinit = tf.glorot_uniform_initializer()
        self.binit = tf.constant_initializer(0.0)

    def initSummaries(self):
        self.inputsSummary = tf.summary.image("raw inputs", self.placeholder.inputs, max_outputs=CONFIG.maxImages)
        self.inputsSummary = tf.summary.image("modified inputs", self.inputs(), max_outputs=CONFIG.maxImages)
        self.outputsSummary = tf.summary.image("outputs", self.outputs(), max_outputs=CONFIG.maxImages)
        self.diffSummary = tf.summary.image("diffs", self.diff, max_outputs=CONFIG.maxImages)
        self.costSummary = tf.summary.scalar("cost", self.cost)
        self.layerSummaries = [tf.summary.histogram("layer/raw_inputs", self.placeholder.inputs)]
        for layer in self.layers:
            self.layerSummaries.append(
                tf.summary.histogram("layer/"+layer.name, layer.outputs))
        weightGradient = tf.gradients(self.cost, self.weights)
        self.weightSummaries = []
        for i in range(len(weightGradient)):
            self.weightSummaries.append(
                tf.summary.histogram("weight/"+str(i), weightGradient[i]))
        biasGradient = tf.gradients(self.cost, self.bias)
        self.biasSummaries = []
        for i in range(len(biasGradient)):
            self.biasSummaries.append(
                tf.summary.histogram("bias/"+str(i), biasGradient[i]))
        self.summaries = tf.summary.merge_all()

    def initPlaceholder(self):
        self.placeholder = types.SimpleNamespace()
        self.placeholder.inputs = tf.placeholder(tf.float32, shape=[None]+self.inputsShape)
        self.batchSize = tf.shape(self.placeholder.inputs)[0]
        self.placeholder.learningRate = tf.placeholder(tf.float32, shape=[])
        self.placeholder.code = tf.placeholder(tf.float32, shape=[None,HYPERPARAMETER.diffLatentSpace])

    def train(self, epoch=HYPERPARAMETER.epoch, dataset=None):
        if dataset is None:
            dataset = self.trainSet

        batchs = []
        start = 0
        while start < dataset.shape[0]:
            end = min(start+HYPERPARAMETER.batchSize, dataset.shape[0])
            batchs.append({"start":start, "end":end})
            start = end

        for i in range(epoch):
            print("EPOCH %d :" % i)
            #np.random.shuffle(dataset)
            for b in tqdm(batchs):
                    trainData = dataset[b["start"]:b["end"]]
                    self.session.run(self.optim, feed_dict={
                        self.placeholder.learningRate: HYPERPARAMETER.learningRate,
                        self.placeholder.inputs: trainData
                    })
            data = dataset[0:CONFIG.maxImages]
            if self.log:
                summaries = self.session.run(self.summaries, feed_dict={
                    self.placeholder.learningRate: HYPERPARAMETER.learningRate,
                    self.placeholder.inputs: data
                })
            if self.log:
                self.writer.add_summary(summaries, i)

    def getDiff(self, dataset):
        diffs = []

        batchs = []
        start = 0
        while start < dataset.shape[0]:
            end = min(start+HYPERPARAMETER.batchSize, dataset.shape[0])
            batchs.append({"start":start, "end":end})
            start = end

        for b in tqdm(batchs):
            data = dataset[b["start"]:b["end"]]
            diff = self.session.run(self.diff, feed_dict={
                self.placeholder.learningRate: HYPERPARAMETER.learningRate,
                self.placeholder.inputs: data
            })
            diffs.append(diff)

        data = dataset[0:CONFIG.maxImages]
        if self.log:
            summaries = self.session.run(self.summaries, feed_dict={
                self.placeholder.learningRate: HYPERPARAMETER.learningRate,
                self.placeholder.inputs: data
            })
        if self.log:
            self.writer.add_summary(summaries)

        return np.concatenate(diffs)

    def encode(self, dataset):
        result = self.session.run(self.encoder, feed_dict={
            self.placeholder.inputs: dataset
        })
        return result

    def generate(self, codes, example):
        result = self.session.run(self.outputs2(), feed_dict={
            self.placeholder.inputs: example,
            self.placeholder.code: codes
        })
        return result

    def reproduce(self, dataset):
        result = self.session.run(self.outputs(), feed_dict={
            self.placeholder.inputs: dataset
        })
        return result


    def outputs(self):
        return self.layers[-1].outputs

    def outputs2(self):
        return self.layers[-1].outputs2

    def inputs(self):
        return self.layers[0].outputs

    def addLayer(self, layer):
        self.layers.append(layer)
        if not layer.layerType in self.layersByType:
            self.layersByType[layer.layerType] = []
        self.layersByType[layer.layerType].append(layer)

    def getWeight(self, name, shape, dtype, initializer):
        w = tf.get_variable(
            name = name,
            shape = shape,
            dtype = dtype,
            initializer = initializer)
        self.weights.append(w)
        return w

    def getBias(self, name, shape, dtype, initializer):
        w = tf.get_variable(
            name = name,
            shape = shape,
            dtype = dtype,
            initializer = initializer)
        self.bias.append(w)
        return w


