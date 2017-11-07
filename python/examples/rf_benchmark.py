# -*- coding: utf-8 -*-
# rf_benchmark.py - Comparison with sklearn
# author : Antoine Passemiers

import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from scythe.core import *
from scythe.plot import plot_feature_importances


def main():
    n_features = 100
    n_samples  = 10000
    n_classes  = 2

    max_n_features = 20
    max_depth = 20
    n_estimators = 10
    min_threshold = 1e-07

    X, y = make_classification(
        n_samples     = 2 * n_samples, 
        n_features    = n_features, 
        n_informative = 7,
        n_classes     = n_classes)

    X_train = X[:n_samples]
    y_train = y[:n_samples]

    X_test = X[n_samples:]
    y_test = y[n_samples:]


    forest = RandomForestClassifier(
        criterion = "gini",
        max_features = max_n_features,
        max_depth = max_depth,
        n_estimators = n_estimators,
        min_impurity_split = min_threshold)
    t0 = time.time()
    forest.fit(X_train, y_train)
    dt_sklearn = time.time() - t0
    predictions = forest.predict(X_test)
    node_counts = list()
    for i, tree in enumerate(forest.estimators_):
        node_counts.append(tree.tree_.node_count)
    sklearn_mnode_count = sum(node_counts)
    sklearn_acc = (predictions == y_test).sum() / float(n_samples)


    fconfig = ForestConfiguration()
    fconfig.n_classes        = n_classes
    fconfig.max_n_trees      = n_estimators
    fconfig.bagging_fraction = 0.4
    fconfig.max_depth        = max_depth - 8
    fconfig.max_n_features   = max_n_features
    fconfig.min_threshold    = min_threshold
    fconfig.partitioning     = 50

    forest = Forest(fconfig, "classification", "rf")
    t0 = time.time()
    forest.fit(X_train, y_train)
    dt_scythe = time.time() - t0

    scythe = Scythe()
    pruning_ids, hs, accs = list(), list(), list()
    for h in range(20, 0, -1):
        pruning_ids.append(scythe.prune_forest_height(forest, h))
        predictions = forest.predict(X_test)
        acc = (predictions.argmax(axis = 1) == y_test).sum() / float(n_samples)
        hs.append(h)
        accs.append(acc)
        scythe.restore(pruning_ids[-1])
    
    scythe.prune(pruning_ids[np.argmax(accs)])
    predictions = forest.predict(X_test)
    scythe_acc = (predictions.argmax(axis = 1) == y_test).sum() / float(n_samples)

    plot_feature_importances(forest, alpha = 1.4)
    plt.show()

    plt.plot(hs, accs)
    plt.show()



    print("Sklearn mean node count : %i" % sklearn_mnode_count)
    print("Scythe time      : %.3f s" % dt_scythe)
    print("Sklearn time     : %.3f s" % dt_sklearn)
    print("Scythe accuracy  : %f" % scythe_acc)
    print("Sklearn accuracy : %f" % sklearn_acc)

if __name__ == "__main__":
    main()