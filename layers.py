import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

countDict = {}

def getCount(name):
    if name not in countDict:
        countDict[name] = 0
    countDict[name] += 1
    return countDict[name]


class Layer(ABC):

    def __init__(self, model, out, shape, activation=None, out2=None):
        self.model = model
        self.shape = shape
        model.addLayer(self)
        self.outputs2 = out2
        if activation is None:
            self.outputs = out
        else:
            self.outputs = activation(out)
            if out2 is not None:
                self.outputs2 = activation(out2)
        print("build : " + self.name + " with shape " + str(model.outputs().shape) + "/" + str(self.shape) + (" *2" if self.outputs2 is not None else ""))

    def checkShape(self, ndim):
        if len(self.shape) != ndim:
            print("wrong shape : %s must be %d dimensions" % (str(self.shape), ndim))
            exit(1)

    def len(self):
        r = 1
        for x in self.shape:
            r *= x
        return r


class InputLayer(Layer):

    layerType = "input"

    def __init__(self, model, placeholder, shape, contrast=None, crop=None, resize=None, normalize=False):
        name = "%sInputs" % model.prefix()
        self.name = name + ("_%d" % getCount(name))
        out = placeholder
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            if crop is not None:
                out = tf.image.crop_to_bounding_box(
                    out,
                    crop[0],
                    crop[1],
                    shape[0] - crop[0] - crop[2],
                    shape[1] - crop[1] - crop[3])
                shape[0] -= (crop[0] + crop[2])
                shape[1] -= (crop[1] + crop[3])
            if resize is not None:
                shape[0] = int((shape[0] * resize) / 100)
                shape[1] = int((shape[1] * resize) / 100)
                out = tf.image.resize_images(out, [shape[0], shape[1]])
            if contrast is not None:
                out = tf.nn.relu(tf.image.adjust_contrast(out, contrast))
            if normalize == "individual":
                print("individual normalization")
                vmax = tf.reduce_max(out, axis=[3,2,1])
                vmin = tf.reduce_min(out, axis=[3,2,1])
                vrange = vmax - vmin
                out = tf.transpose((tf.transpose(out) - vmin) / vrange)
            elif normalize == "normal":
                print("default normalization")
                vmax = tf.reduce_max(out)
                vmin = tf.reduce_min(out)
                vrange = vmax - vmin
                out = (out-vmin)/vrange
        Layer.__init__(self, model, out, shape)

class ConvutionalLayer(Layer):

    layerType = "convutional"

    def __init__(self, model, filterShape, stride, activation = None):
        name = "%sConv" % model.prefix()
        self.name = name + ("_%d" % getCount(name))
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            w = model.getWeight(
                name = "weights",
                shape = filterShape,
                dtype = tf.float32,
                initializer = model.convinit)
            b = model.getBias(
                name = "bias",
                shape = [filterShape[3]],
                dtype = tf.float32,
                initializer = model.binit)
            out = tf.nn.conv2d(
                model.outputs(),
                w,
                stride,
                name=self.name,
                padding="SAME")
            out += b
        lastShape = model.layers[-1].shape
        ch = filterShape[3]
        w = np.ceil(lastShape[1] / stride[2])
        h = np.ceil(lastShape[0] / stride[1])
        Layer.__init__(self, model, out, [h,w,ch], activation)
        self.checkShape(3)

class DeconvutionalLayer(Layer):

    layerType = "deconvutional"

    def __init__(self, model, filterShape, outputShape, stride, activation = None):
        name = "%sDeconv" % model.prefix()
        self.name = name + ("_%d" % getCount(name))
        out2 = None
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            w = model.getWeight(
                name = "weights",
                shape = filterShape,
                dtype = tf.float32,
                initializer = model.convinit)
            b = model.getBias(
                name = "bias",
                shape = [1],
                dtype = tf.float32,
                initializer = model.binit)
            out = tf.nn.conv2d_transpose(
                model.outputs(),
                w,
                outputShape,
                stride,
                name=self.name,
                padding="SAME")
            out += b
            if model.outputs2() is not None:
                out2 = tf.nn.conv2d_transpose(
                    model.outputs2(),
                    w,
                    outputShape,
                    stride,
                    name=self.name,
                    padding="SAME")
                out2 += b
        Layer.__init__(self, model, out, outputShape[1:], activation, out2)
        self.checkShape(3)


class Dense(Layer):

    layerType = "dense"

    def __init__(self, model, outputLen, activation = None):
        name = "%sDense" % model.prefix()
        self.name = name + ("_%d" % getCount(name))
        out2 = None
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            w = model.getWeight(
                name = "weights",
                shape = [model.outputs().shape[1], outputLen],
                dtype = tf.float32,
                initializer = model.winit)
            b = model.getBias(
                name = "bias",
                shape = [outputLen],
                dtype = tf.float32,
                initializer = model.binit)
            mm = tf.matmul(model.outputs(), w)
            out = mm + b
            if model.outputs2() is not None:
                mm2 = tf.matmul(model.outputs2(), w)
                out2 = mm2 + b
        Layer.__init__(self, model, out, [outputLen], activation, out2)
        self.checkShape(1)

class Dense(Layer):

    layerType = "dense"

    def __init__(self, model, outputLen, activation = None, factor=None):
        name = "%sDense" % model.prefix()
        self.name = name + ("_%d" % getCount(name))
        out2 = None
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            w = model.getWeight(
                name = "weights",
                shape = [model.outputs().shape[1], outputLen],
                dtype = tf.float32,
                initializer = model.winit)
            b = model.getBias(
                name = "bias",
                shape = [outputLen],
                dtype = tf.float32,
                initializer = model.binit)
            mm = tf.matmul(model.outputs(), w)
            out = mm + b
            if factor is not None:
                out = factor * out
            if model.outputs2() is not None:
                mm2 = tf.matmul(model.outputs2(), w)
                out2 = mm2 + b
                if factor is not None:
                    out2 = factor * out2
        Layer.__init__(self, model, out, [outputLen], activation, out2)
        self.checkShape(1)

class VAE(Layer):

    layerType = "VAE"

    def __init__(self, model, mean, std):
        name = "%sVAE" % model.prefix()
        self.name = name + ("_%d" % getCount(name))
        out2 = None
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            epsilon = tf.random_normal(tf.stack([model.batchSize, mean.shape[0]]))
            out = mean.outputs + tf.multiply(epsilon, tf.exp(std.outputs))
            if mean.outputs2 is not None:
                if std.outputs2 is not None:
                    out2 = mean.outputs2 + tf.multiply(epsilon, tf.exp(std.outputs2))
                else:
                    out2 = mean.outputs2
        Layer.__init__(self, model, out, [mean.shape[0]], out2=out2)
        self.checkShape(1)

class NAC(Layer):

    layerType = "NAC"

    def __init__(self, model, outputLen):
        name = "%sNAC" % model.prefix()
        self.name = name + ("_%d" % getCount(name))
        out2 = None
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            w = model.getWeight(
                name = "weights",
                shape = [model.outputs().shape[1], outputLen],
                dtype = tf.float32,
                initializer = model.winit)
            m = model.getWeight(
                name = "weights",
                shape = [model.outputs().shape[1], outputLen],
                dtype = tf.float32,
                initializer = model.winit)
            W = tf.nn.tanh(w) * tf.nn.sigmoid(m)
            out = tf.matmul(model.outputs(), W)
            if model.outputs2() is not None:
                out2 = tf.matmul(model.outputs2(), W)
        Layer.__init__(self, model, out, [outputLen], None, out2)
        self.checkShape(1)

class Reshape(Layer):

    layerType = "reshape"
    def __init__(self, model, newShape):
        name = "%sReshape" % model.prefix()
        self.name = name + ("_%d" % getCount(name))
        out = tf.reshape(model.outputs(), newShape)
        out2 = None
        if model.outputs2() is not None:
            out2 = tf.reshape(model.outputs2(), newShape)
        shape = newShape[1:]
        Layer.__init__(self, model, out, shape, None, out2)
