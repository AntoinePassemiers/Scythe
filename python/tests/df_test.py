# -*- coding: utf-8 -*-
# df_test.py: Tests on the deep forest
# author : Antoine Passemiers

import os, sys

import matplotlib.pyplot as plt

from scythe.core import *
from scythe.layers import *

def minimal_test():
    n_forests_per_layer = 2
    kc, kr = 2, 2

    fconfig = ForestConfiguration()
    # fconfig.bag_size       = 60000
    fconfig.n_classes      = 2
    fconfig.max_n_trees    = 2
    fconfig.max_n_features = 20
    fconfig.max_depth      = 20
    lconfig = LayerConfiguration(fconfig, n_forests_per_layer, COMPLETE_RANDOM_FOREST)

    print("Create gcForest")
    graph = DeepForest(task = "classification", n_classes = 2)
    scanner = MultiGrainedScanner2D(lconfig, (kc, kr))
    graph.add(scanner)
    cascade = CascadeLayer(lconfig)
    graph.add(cascade)

    cascade2 = CascadeLayer(lconfig)
    graph.add(cascade2)


    X_train = np.array(
        [[[25, 20, 15], [5, 0, 0], [0, 1, 0]],
        [[0, 4, 2], [5, 0, 15], [20, 42, 15]]], dtype = np.uint8)
    y_train = np.array([0, 1])
    X_test, labels = X_train, y_train

    print("Fit gcForest")
    graph.fit(X_train, y_train)

    print("Classify with gcForest")
    probas = graph.classify(X_test)
    print(probas)
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
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    minimal_test()