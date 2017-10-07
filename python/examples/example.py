# -*- coding: utf-8 -*-
# example.py : Scythe example of use on MNIST dataset
# author : Antoine Passemiers

import os, sys

from scythe.core import *
from scythe.layers import *
from scythe.MNIST import *

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
    if len(sys.argv) < 2:
        print("Please provide the path to the MNIST dataset as first argument")
        return
    mnist_folder = sys.argv[1]

    n_forests_per_layer = 2
    kc, kr = 22, 22

    fconfig = ForestConfiguration()
    fconfig.bag_size       = 60000
    fconfig.n_classes      = 10
    fconfig.max_n_trees    = 4
    fconfig.max_n_features = 20
    fconfig.max_depth      = 12
    lconfig = LayerConfiguration(fconfig, n_forests_per_layer, COMPLETE_RANDOM_FOREST)

    print("Create gcForest")
    graph = DeepForest(task = "classification", n_classes = 10)

    print("Add 2D Convolutional layer")
    scanner = MultiGrainedScanner2D(lconfig, (kc, kr))
    scanner_id = graph.add(scanner)

    print("Add cascade layer")
    cascade = CascadeLayer(lconfig)
    cascade_id = graph.add(cascade)

    print("Add cascade layer")
    cascade2 = CascadeLayer(lconfig)
    cascade2_id = graph.add(cascade2)

    graph.connect(scanner_id, cascade2_id)


    X_train, y_train = loadMNISTTrainingSet(location = mnist_folder)
    X_test, labels = loadMNISTTestSet(location = mnist_folder)
    X_train, y_train = X_train[:100], y_train[:100]
    # X_test, labels = X_train, y_train # TO REMOVE
    # X_test, labels = X_test[:100, :], labels[:100]

    print("Fit gcForest")
    graph.fit(X_train, y_train)

    print("Classify with gcForest")
    probas = graph.classify(X_test)
    predictions = probas.argmax(axis = 1)
    ga = np.sum(predictions == labels)
    print("Correct predictions : %i / %i" % (ga, len(labels)))

    f = scanner.getForests()
    feature_importances = f[0].getFeatureImportances()
    feature_importances = feature_importances.reshape(kc, kr)

    plt.imshow(feature_importances)
    plt.title("Feature importances (receptive field)")
    plt.show()

if __name__ == "__main__":
    main()