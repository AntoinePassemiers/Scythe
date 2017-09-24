# -*- coding: utf-8 -*-
# encoders.py : Dataset preprocessing and speed comparison
# author : Antoine Passemiers

import random, timeit
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scythe.encoders import HashEncoder

"""
Comparison between Scythe's HashEncoder and the two main alternatives
for converting strings to integers.
Sklearn's LabelEncoder is slower by nature because it guarantees that
output values fall in the range [0, n_classes], where n_classes is the
number of classes to consider in the case of a classification task.

For encoding target values (labels), one should use sklearn.preprocessing.LabelEncoder
For encoding explanatory variables, one should use scythe.encoders.HashEncoder
"""

if __name__ == "__main__":
    n_samples = 100000
    CHARACTERS = "abcdefghijklmnopqrstuwxyz"
    values = [''.join(random.sample(CHARACTERS, 10)) for i in range(n_samples)]

    print("Running tests on %i samples..." % n_samples)

    t0 = timeit.default_timer()
    encoder = HashEncoder(dtype = np.int32)
    d = encoder.encode(values)
    print("Scythe's HashEncoder: %s s" % str(timeit.default_timer() - t0))

    t0 = timeit.default_timer()
    encoder = LabelEncoder()
    d = encoder.fit_transform(values)
    print("Sklearn's LabelEncoder: %s s" % str(timeit.default_timer() - t0))

    t0 = timeit.default_timer()
    dataframe = pd.DataFrame({"values" : values})
    d = dataframe["values"].apply(hash)
    print("Pandas + Python default hash function: %s s" % str(timeit.default_timer() - t0))

    print("Finished")