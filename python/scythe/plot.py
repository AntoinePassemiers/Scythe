# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importances(predictor, alpha = 2.0):
    feature_importances = predictor.getFeatureImportances()
    std_dev = np.std(feature_importances)
    bar_list = plt.bar(np.arange(len(feature_importances)), feature_importances)
    for (bar_item, importance) in zip(bar_list, feature_importances):
    	if importance > alpha * std_dev:
    		bar_item.set_color("orangered")
    	else:
    		bar_item.set_color("orange")
    plt.title("Feature importances")
    plt.xlabel("Feature id")
    plt.ylabel("Weighted information gain")

def show():
	plt.show()