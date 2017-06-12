# -*- coding: utf-8 -*-
# example.py : Scythe example of use
# author : Antoine Passemiers

from core import * # TODO
from layers import * # TODO

if __name__ == "__main__":
    forest = DeepForest(task = "classification")

    forest.add(MultiGrainedScanner1D())
    # forest.add(DirectLayer())
    # forest.add(CascadeLayer())
    # forest.add(CascadeLayer())

    print("Finished")