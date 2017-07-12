# -*- coding: utf-8 -*-
# rf_benchmark.py - Comparison with sklearn
# author : Antoine Passemiers

from core import * # TODO

import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_splitter.pyx

if __name__ == "__main__":

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
    fconfig.bag_size  = 1000
    fconfig.max_height = max_depth
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


    print("Finished")