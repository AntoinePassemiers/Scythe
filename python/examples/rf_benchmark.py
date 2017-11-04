# -*- coding: utf-8 -*-
# rf_benchmark.py - Comparison with sklearn
# author : Antoine Passemiers

from scythe.core import *
from scythe.plot import plot_feature_importances

import time

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt


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

    fconfig = ForestConfiguration()
    fconfig.n_classes        = n_classes
    fconfig.max_n_trees      = n_estimators
    fconfig.bagging_fraction = 1.0
    fconfig.max_depth        = max_depth - 8
    fconfig.max_n_features   = max_n_features
    fconfig.min_threshold    = min_threshold

    forest = Forest(fconfig, "classification", "rf")
    t0 = time.time()
    forest.fit(X_train, y_train)
    dt_scythe = time.time() - t0

    scythe = Scythe()
    pruning_ids, hs, losses = list(), list(), list()
    for h in range(20, 0, -1):
        pruning_ids.append(scythe.prune_forest_height(forest, h))
        predictions = forest.predict(X_test)
        loss = log_loss(y_test, predictions)
        hs.append(h)
        losses.append(loss)
        scythe.restore(pruning_ids[-1])
    
    # scythe.prune(pruning_ids[np.argmin(losses)])
    predictions = forest.predict(X_test)
    scythe_acc = (predictions.argmax(axis = 1) == y_test).sum() / float(n_samples)

    plot_feature_importances(forest, alpha = 1.4)
    plt.show()

    plt.plot(hs, losses)
    plt.show()


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
