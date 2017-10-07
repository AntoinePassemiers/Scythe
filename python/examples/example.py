# -*- coding: utf-8 -*-
# example.py : Scythe example of use on MNIST dataset
# author : Antoine Passemiers

import os, sys

from scythe.core import *
from scythe.layers import *
from scythe.MNIST import *

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Please provide the path to the MNIST dataset as first argument")
        return
    mnist_folder = sys.argv[1]

    n_forests_per_layer = 2
    kc, kr = 22, 22

    fconfig = ForestConfiguration()
    fconfig.n_classes      = 10
    fconfig.max_n_trees    = 50
    fconfig.max_n_features = 20
    fconfig.max_depth      = 8
    lconfig = LayerConfiguration(fconfig, n_forests_per_layer, COMPLETE_RANDOM_FOREST)

    print("Create gcForest")
    graph = DeepForest(task = "classification", n_classes = 10)

    # scanner is set as both front layer and rear layer
    scanner = MultiGrainedScanner2D(lconfig, (kc, kr))
    scanner_id = graph.add(scanner)

    # cascade is added to rear's chidren (scanner)
    # cascade is then set as rear layer
    cascade = CascadeLayer(lconfig)
    cascade_id = graph.add(cascade)

    # cascade2 is added to rear's children (cascade)
    # cascade2 is then set as rear layer
    cascade2 = CascadeLayer(lconfig)
    cascade2_id = graph.add(cascade2)

    # connect scanner and cascade2
    graph.connect(scanner_id, cascade2_id)

    print("Load MNIST datasets")
    X_train, y_train = loadMNISTTrainingSet(location = mnist_folder)
    X_test, labels = loadMNISTTestSet(location = mnist_folder)

    # Scythe's deep forest is still relatively slow
    # Better use a subset of the training set for experimenting purposes
    X_train, y_train = X_train[:1000], y_train[:1000]

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