# -*- coding: utf-8 -*-
# example.py : Scythe example of use
# author : Antoine Passemiers

from core import * # TODO
from layers import * # TODO

if __name__ == "__main__":

    fconfig = ForestConfig(
        n_classes   =  3,
        n_iter      = 50,
        max_n_trees = 10,
        max_depth   = 6)
    lconfig = LayerConfig(fconfig, 3, COMPLETE_RANDOM_FOREST)

    print("Create gcForest")
    graph = DeepForest(task = "classification")
    
    print("Add layer")
    graph.add(MultiGrainedScanner1D(lconfig, (3,)))
    # graph.add(DirectLayer())
    # graph.add(CascadeLayer())
    # graph.add(CascadeLayer())

    X = MDDataset(np.random.rand(800, 10))
    y = Labels(np.random.randint(0, 3, size = 800))

    print("Fit gcForest")
    graph.fit(X, y)

    print("Finished")