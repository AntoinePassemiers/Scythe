# -*- coding: utf-8 -*-
# rf_benchmark.py - Comparison with sklearn
# author : Antoine Passemiers

from scythe.core import * # TODO

import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Debugging : gdb -q -x rf_benchmark.py

def main():
    n_features = 100
    n_samples  = 10000
    n_classes  = 2

    max_n_features = 20
    max_depth = 20
    n_estimators = 10

    X, y = make_classification(
        n_samples  = 2 * n_samples, 
        n_features = n_features, 
        n_classes  = n_classes)

    X_train = Dataset(X[:n_samples])
    y_train = Labels(y[:n_samples])

    X_test = Dataset(X[n_samples:])
    y_test = Labels(y[n_samples:])    

    fconfig = ForestConfig()
    fconfig.n_classes = n_classes
    fconfig.n_iter    = n_estimators
    fconfig.bag_size  = 10000
    fconfig.max_depth = max_depth
    fconfig.max_n_features = max_n_features

    forest = Forest(fconfig, "classification", "random forest")
    forest.fit(X_train, y_train)
    predictions = forest.predict(X_test)
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

    tmp = X[:n_samples].reshape(n_samples, 20, 20)
    X_train = MDDataset(tmp)
    y_train = Labels(y[:n_samples])

    tmp = X[n_samples:].reshape(n_samples, 20, 20)
    X_test = MDDataset(tmp)
    y_test = Labels(y[n_samples:])    

    fconfig = ForestConfig()
    fconfig.n_classes = n_classes
    fconfig.n_iter    = 5
    fconfig.bag_size  = 1000
    fconfig.max_depth = 10
    fconfig.max_n_features = max_n_features
    lconfig = LayerConfig(fconfig, 2, COMPLETE_RANDOM_FOREST)

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