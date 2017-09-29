# -*- coding: utf-8 -*-
# example.py : Scythe example of use
# author : Antoine Passemiers

import os, sys

from scythe.core import *
from scythe.layers import *

from scythe.MNIST import *  # TODO

import matplotlib.pyplot as plt


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

    fconfig = ForestConfiguration()
    fconfig.bag_size       = 60000
    fconfig.n_classes      = 10
    fconfig.max_n_trees    = 4
    fconfig.max_n_features = 20
    fconfig.max_depth      = 20
    lconfig = LayerConfiguration(fconfig, n_forests_per_layer, COMPLETE_RANDOM_FOREST)

    print("Create gcForest")
    graph = DeepForest(task = "classification", n_classes = 10)

    print("Add 2D Convolutional layer")
    scanner = MultiGrainedScanner2D(lconfig, (kc, kr))
    graph.add(scanner)

    #print("Add cascade layer")
    #cascade = CascadeLayer(lconfig)
    #graph.add(cascade)

    print("Add cascade layer")
    cascade = CascadeLayer(lconfig)
    graph.add(cascade)

    X_train, y_train = loadMNISTTrainingSet(location = sys.argv[1])
    X_test, labels = loadMNISTTestSet(location = sys.argv[1])
    X_train, y_train = X_train[:500], y_train[:500]
    X_test, labels = X_train, y_train # TO REMOVE

    print("Fit gcForest")
    graph.fit(X_train, y_train)

    print("Classify with gcForest")
    probas = graph.classify(X_test)
    predictions = probas.argmax(axis = 1)
    ga = np.sum(predictions == labels)
    print(predictions)
    print(labels)
    print("Correct predictions : %i / %i" % (ga, len(labels)))

    f = scanner.getForests()
    feature_importances = f[0].getFeatureImportances()
    feature_importances = feature_importances.reshape(kc, kr)

    plt.imshow(feature_importances)
    plt.title("Feature importances")
    plt.show()


def testRF():
    X_train, y_train = loadMNISTTrainingSet(location = sys.argv[1])
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

    X_test, labels = loadMNISTTestSet(location = sys.argv[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    n_samples = len(labels)

    fconfig = ForestConfiguration()
    fconfig.n_classes = 10
    fconfig.max_n_trees = 50
    fconfig.bag_size  = 60000
    fconfig.max_depth = 20
    fconfig.max_n_features = 50

    forest = Forest(fconfig, "classification", "rf")
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
