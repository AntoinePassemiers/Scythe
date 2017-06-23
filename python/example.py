# -*- coding: utf-8 -*-
# example.py : Scythe example of use
# author : Antoine Passemiers

import os, sys

from core import *   # TODO
from layers import * # TODO
from MNIST import *  # TODO


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Warning: provide the path to the MNIST dataset")

    fconfig = ForestConfig(
        n_classes   = 10,
        n_iter      = 50,
        max_n_trees = 10,
        max_depth   = 6)
    lconfig = LayerConfig(fconfig, 3, COMPLETE_RANDOM_FOREST)

    print("Create gcForest")
    graph = DeepForest(task = "classification")
    
    print("Add 2D Convolutional layer")
    graph.add(MultiGrainedScanner2D(lconfig, (10, 10)))

    print("Add cascade layer")
    graph.add(CascadeLayer(lconfig))
    # graph.add(CascadeLayer())
    
    X_train, y_train = loadMNISTTrainingSet(location = sys.argv[1])
    X_train, y_train = MDDataset(X_train), Labels(y_train)

    print("Fit gcForest")
    graph.fit(X_train, y_train)

    X_test, y_test = loadMNISTTestSet(location = sys.argv[1])
    X_test, y_test = MDDataset(X_test), Labels(y_test)

    print("Classify with gcForest")
    probas = graph.classify(X_test)
    print(probas)

    print("Finished")