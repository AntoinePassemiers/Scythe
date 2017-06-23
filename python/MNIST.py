# -*- coding: utf-8 -*-
# MNIST.py : Load the MNIST dataset
# author : Antoine Passemiers

import os, sys, struct
import numpy as np


def loadMNISTTrainingSet(location = "."):

    y_filepath = os.path.join(location, "train-labels-idx1-ubyte")
    with open(y_filepath, "rb") as y_file:
        _, _ = struct.unpack(">II", y_file.read(8))
        labels = np.fromfile(y_file, dtype = np.int8)

    X_filepath = os.path.join(location, "train-images-idx3-ubyte")
    with open(X_filepath, "rb") as X_file:
        _, _, instances, features = struct.unpack(">IIII", X_file.read(16))
        images = np.fromfile(
            X_file, dtype = np.uint8).reshape(
                len(labels), instances, features)
    print("MNIST training set loaded: %i images" % len(images))
    return images, labels


def loadMNISTTestSet(location = "."):

    y_filepath = os.path.join(location, "t10k-labels-idx1-ubyte")
    with open(y_filepath, "rb") as y_file:
        _, _ = struct.unpack(">II", y_file.read(8))
        labels = np.fromfile(y_file, dtype = np.int8)

    X_filepath = os.path.join(location, "t10k-images-idx3-ubyte")
    with open(X_filepath, "rb") as X_file:
        _, _, instances, features = struct.unpack(">IIII", X_file.read(16))
        images = np.fromfile(
            X_file, dtype = np.uint8).reshape(
                len(labels), instances, features)

    return images, labels