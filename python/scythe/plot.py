# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importances(predictor):
    feature_importances = predictor.getFeatureImportances()
    plt.bar(np.arange(len(feature_importances)), feature_importances)
    plt.title("Feature importances")
    plt.xlabel("Feature id")
    plt.ylabel("Weighted information gain")

def show():
	plt.show()