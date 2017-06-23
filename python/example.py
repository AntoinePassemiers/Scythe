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
        n_classes   =  3,
        n_iter      = 50,
        max_n_trees = 10,
        max_depth   = 6)
    lconfig = LayerConfig(fconfig, 3, COMPLETE_RANDOM_FOREST)

    print("Create gcForest")
    graph = DeepForest(task = "classification")
    
    print("Add layer")
    graph.add(MultiGrainedScanner2D(lconfig, (10, 10)))
    # graph.add(DirectLayer())
    # graph.add(CascadeLayer())
    # graph.add(CascadeLayer())
    
    X, y = loadMNISTTrainingSet(location = sys.argv[1])
    X, y = MDDataset(X), Labels(y)

    print("Fit gcForest")
    graph.fit(X, y)

    print("Finished")