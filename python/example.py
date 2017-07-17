# -*- coding: utf-8 -*-
# example.py : Scythe example of use
# author : Antoine Passemiers

import os, sys

from core import *   # TODO
from layers import * # TODO
from MNIST import *  # TODO


"""
Hyper-parameters (as determined by Zhi-Hua Zhou and Ji Feng)
----------------

Multi-grained scanner:
    Number of forests: 2
    Number of trees per forest: 500
    Maximum depth: 100
Cascade layer:
    Number of forests: 8
    Number of trees per forest: 500
    Maximum depth: infinite
"""

def main():
    n_forests_per_layer = 2
    kc, kr = 22, 22
    required_nbytes = MultiGrainedScanner2D.estimateRequiredBufferSize(
        60000, 28, 28, kc, kr, 10, n_forests_per_layer) * 8
    nbytes = bytesToStr(required_nbytes)
    print("Required number of bytes in the convolutional layer : %s" % nbytes)

    fconfig = ForestConfig(
        bag_size       = 60000,
        n_classes      = 10,
        n_iter         = 4,
        max_n_trees    = 4,
        max_n_features = 20,
        max_depth      = 4)
    lconfig = LayerConfig(fconfig, n_forests_per_layer, COMPLETE_RANDOM_FOREST)

    X_test, y_test = loadMNISTTestSet(location = sys.argv[1])

    print("Create gcForest")
    graph = DeepForest(task = "classification")

    print("Add 2D Convolutional layer")
    graph.add(MultiGrainedScanner2D(lconfig, (kc, kr)))

    print("Add cascade layer")
    graph.add(CascadeLayer(lconfig))

    X_train, y_train = loadMNISTTrainingSet(location = sys.argv[1])
    X_train, y_train = MDDataset(X_train), Labels(y_train)

    print("Fit gcForest")
    graph.fit(X_train, y_train)

    X_test, labels = loadMNISTTestSet(location = sys.argv[1])
    X_test, y_test = MDDataset(X_test), Labels(y_test)

    print("Classify with gcForest")
    probas = graph.classify(X_test)
    predictions = probas.argmax(axis = 1)
    ga = np.sum(predictions == labels)
    print(list(predictions))
    print("Correct predictions : %i / %i" % (ga, len(labels)))


def testRF():
    X_train, y_train = loadMNISTTrainingSet(location = sys.argv[1])
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_train, y_train = Dataset(X_train), Labels(y_train)

    X_test, labels = loadMNISTTestSet(location = sys.argv[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    X_test, y_test = Dataset(X_test), Labels(labels)

    n_samples = len(labels)

    fconfig = ForestConfig()
    fconfig.n_classes = 10
    fconfig.n_iter    = 4
    fconfig.bag_size  = 60000
    fconfig.max_depth = 20
    fconfig.max_n_features = 50

    forest = Forest(fconfig, "classification", "random forest")
    forest.fit(X_train, y_train)

    probabilities = forest.predict(X_test)
    predictions = probabilities.argmax(axis = 1)

    accuracy = (predictions == labels).sum() / float(n_samples)
    print(accuracy)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Warning: provide the path to the MNIST dataset")

    # testRF()
    main()

    print("Finished")
