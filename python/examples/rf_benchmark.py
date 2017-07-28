# -*- coding: utf-8 -*-
# rf_benchmark.py - Comparison with sklearn
# author : Antoine Passemiers

from scythe.core import *

import time

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def main():
    n_features = 100
    n_samples  = 10000
    n_classes  = 2

    max_n_features = 20
    max_depth = 20
    n_estimators = 10
    min_threshold = 1e-07

    X, y = make_classification(
        n_samples  = 2 * n_samples, 
        n_features = n_features, 
        n_classes  = n_classes)

    X_train = X[:n_samples]
    y_train = y[:n_samples]

    X_test = X[n_samples:]
    y_test = y[n_samples:]

    fconfig = ForestConfiguration()
    fconfig.n_classes      = n_classes
    fconfig.max_n_trees    = n_estimators
    fconfig.bag_size       = 10000
    fconfig.max_depth      = max_depth - 8
    fconfig.max_n_features = max_n_features
    fconfig.min_threshold  = min_threshold

    forest = Forest(fconfig, "classification", "rf")
    t0 = time.time()
    forest.fit(X_train, y_train)
    predictions = forest.predict(X_test)
    dt_scythe = time.time() - t0
    scythe_acc = (predictions.argmax(axis = 1) == y[n_samples:]).sum() / float(n_samples)

    forest = RandomForestClassifier(
        criterion = "gini",
        max_features = max_n_features,
        max_depth = max_depth,
        n_estimators = n_estimators,
        min_impurity_split = min_threshold)
    t0 = time.time()
    forest.fit(X[:n_samples], y[:n_samples])
    predictions = forest.predict(X[n_samples:])
    dt_sklearn = time.time() - t0
    node_counts = list()
    for i, tree in enumerate(forest.estimators_):
        node_counts.append(tree.tree_.node_count)
    sklearn_mnode_count = sum(node_counts)
    sklearn_acc = (predictions == y[n_samples:]).sum() / float(n_samples)

    print("Sklearn mean node count : %i" % sklearn_mnode_count)
    print("Scythe time      : %.3f s" % dt_scythe)
    print("Sklearn time     : %.3f s" % dt_sklearn)
    print("Scythe accuracy  : %f" % scythe_acc)
    print("Sklearn accuracy : %f" % sklearn_acc)

def convForest():
    n_features = 400
    n_samples  = 1000
    n_classes  = 2

    max_n_features = 20
    max_depth = 5
    n_estimators = 4

    X, y = make_classification(
        n_samples  = 2 * n_samples, 
        n_features = n_features, 
        n_classes  = n_classes)

    X_train = X[:n_samples]
    y_train = y[:n_samples]

    X_test = X[n_samples:]
    y_test = y[n_samples:]  

    fconfig = ForestConfiguration()
    fconfig.n_classes      = n_classes
    fconfig.max_n_trees    = 5
    fconfig.bag_size       = 1000
    fconfig.max_depth      = 30
    fconfig.max_n_features = max_n_features
    lconfig = LayerConfiguration(fconfig, 2, RANDOM_FOREST)


    forest = DeepForest(task = "classification")
    forest.add(MultiGrainedScanner2D(lconfig, (10, 10)))
    forest.add(CascadeLayer(lconfig))
    # forest.add(CascadeLayer(lconfig))


    forest.fit(X_train, y_train)
    print("Deep forest grown")
    predictions = forest.classify(X_test)
    print(predictions.argmax(axis = 1))
    scythe_acc = (predictions.argmax(axis = 1) == y[n_samples:]).sum() / float(n_samples)

    forest = RandomForestClassifier(
        criterion = "gini",
        max_features = max_n_features,
        max_depth = max_depth,
        n_estimators = n_estimators)
    forest.fit(X[:n_samples], y[:n_samples])
    predictions = forest.predict(X[n_samples:])
    for i, tree in enumerate(forest.estimators_):
        print(tree.tree_.node_count)
    sklearn_acc = (predictions == y[n_samples:]).sum() / float(n_samples)

    print("Scythe accuracy  : %f" % scythe_acc)
    print("Sklearn accuracy : %f" % sklearn_acc)



if __name__ == "__main__":
    # convForest()
    main()
    print("Finished")
