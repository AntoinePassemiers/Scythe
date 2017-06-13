# -*- coding: utf-8 -*-
# example.py : Scythe example of use
# author : Antoine Passemiers

from core import * # TODO
from layers import * # TODO

if __name__ == "__main__":

    graph = DeepForest(task = "classification")

    graph.add(MultiGrainedScanner2D())
    # graph.add(DirectLayer())
    # graph.add(CascadeLayer())
    # graph.add(CascadeLayer())

    print("Finished")