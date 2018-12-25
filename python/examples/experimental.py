# -*- coding: utf-8 -*-
# experimental.py
# author : Antoine Passemiers

from scythe.core import ConvForest

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    mnist = fetch_mldata("MNIST original")
    X = mnist.data.reshape((mnist.data.shape[0], 28, 28, 1)) # New shape: (None, 28, 28, 1)
    y = np.reshape(mnist.target, (mnist.target.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./7.)

    conv = ConvForest(3, 3, 1, 2, max_n_nodes=10)
    conv.fit(X_train, y_train)

    print(X_train.max())