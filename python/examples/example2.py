# -*- coding: utf-8 -*-
# example.py : Scythe example of use
# author : Antoine Passemiers

from scythe.core import *

if __name__ == "__main__":

    # TREE CONFIGURATION
    config = TreeConfiguration()
    config.is_incremental = False
    config.min_threshold = 1e-06
    config.max_height = 50
    config.n_classes = 3
    config.max_nodes = 30
    config.partitioning = PERCENTILE_PARTITIONING
    config.nan_value = -1.0
    
    X_train = np.asarray(np.array([
        [0, 0, 0], # 0    1    5.6   6.65  0.5 0.0 0.5
        [0, 0, 1], # 0    0    7.8   7.8   1.0 0.0 0.0
        [1, 0, 0], # 1    1    4.2   4.2   0.0 1.0 0.0
        [2, 0, 0], # 1    1    3.5   3.5   0.0 1.0 0.0
        [2, 1, 0], # 2    1.5  9.8   7.9   0.0 0.5 0.5
        [2, 1, 1], # 0    0    5.4   5.4   1.0 0.0 0.0
        [1, 1, 1], # 1    1    2.1   2.1   0.0 1.0 0.0
        [0, 0, 0], # 2    1    7.7   6.65  0.5 0.0 0.5
        [0, 1, 0], # 2    2    8.8   8.8   0.0 0.0 1.0
        [2, 1, 0], # 1    1.5  6.0   7.9   0.0 0.5 0.5
        [0, 1, 1], # 1    1    5.7   5.7   0.0 1.0 0.0
        [1, 0, 1], # 1    1    7.0   7.0   0.0 1.0 0.0
        [1, 1, 0], # 1    1    6.9   6.9   0.0 1.0 0.0
        [2, 0, 1]  # 0    0    6.3   6.3   1.0 0.0 0.0
    ]), dtype = np.double)
    # X_train = np.random.rand(14, 3)

    y_train = np.array([0, 0, 1, 1, 2, 0, 1, 2, 2, 1, 1, 1, 1, 0])
    X_test = X_train

    # CLASSIFICATION TREE
    tree = Tree(config, "classification")
    tree.fit(X_train, y_train)
    preds = tree.predict(X_test)
    print("\n%s" % preds)

    """
    # REGRESSION TREE
    targets = np.array([5.6, 7.8, 4.2, 3.5, 9.8, 5.4, 2.1, 7.7, 8.8, 6.0, 5.7, 7.0, 6.9, 6.3])
    targets  = Labels(targets)
    tree = Tree(config, "regression")
    tree.fit(dataset, targets)
    preds = tree.predict(testset)
    print("\n%s" % preds)

    # RANDOM FOREST (CLASSIFICATION)
    n_instances = 1000
    n_features = 10
    dataset = Dataset(np.random.randint(0, 2, size = (n_instances, n_features)))
    labels  = Labels(np.random.randint(3, size = n_instances))
    fconfig = ForestConfig()
    fconfig.n_classes = 3
    fconfig.max_depth = 4
    fconfig.max_n_nodes = 500
    fconfig.nan_value = -1.0
    fconfig.n_iter    = 5
    fconfig.learning_rate = 0.05

    forest = Forest(fconfig, "classification", "random forest")
    forest.fit(dataset, labels)
    forest.predict(dataset)

    # COMPLETE RANDOM FOREST (CLASSIFICATION)
    forest = Forest(fconfig, "classification", "complete random forest")
    forest.fit(dataset, labels)
    forest.predict(dataset)
    """

    print("Finished")
