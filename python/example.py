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

    graph = DeepForest(task = "classification")

    graph.add(MultiGrainedScanner1D(lconfig, (3,)))
    # graph.add(DirectLayer())
    # graph.add(CascadeLayer())
    # graph.add(CascadeLayer())

    X = MDDataset(np.random.rand(800, 10))
    y = Labels(np.random.randint(0, 3, size = 800))

    graph.fit(X, y)

    print("Finished")