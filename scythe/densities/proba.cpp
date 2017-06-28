/**
    proba.cpp
    Compute standard probability densities

    @author Antoine Passemiers
    @version 1.0 25/06/2017
*/

#include "proba.hpp"

Density* getArbitraryProbaDensities(size_t n_features, size_t n_classes) {
    assert(n_features > 0);
    Density* densities = new Density[n_features];
    densities[0].quartiles = new data_t[4];
    densities[0].deciles = new data_t[10];
    densities[0].percentiles = new data_t[100];
    densities[0].counters_left = new size_t[n_classes];
    densities[0].counters_right = new size_t[n_classes];
    densities[0].counters_nan = new size_t[n_classes];
    densities[0].values = new data_t[100];
    densities[0].n_values = 100;

    for (unsigned int i = 1; i <= 4; i++) {
        densities[0].quartiles[i - 1] =  i / 4.0;
    }
    for (unsigned int i = 1; i <= 10; i++) {
        densities[0].deciles[i - 1] = i / 10.0;
    }
    for (unsigned int i = 1; i <= 100; i++) {
        densities[0].percentiles[i - 1] = i / 100.0;
    }
    for (unsigned int i = 1; i <= 100; i++) {
        densities[0].values[i - 1] = i / 100.0;
    }

    for (unsigned int f = 1; f < n_features; f++) {
        densities[f] = densities[0];
    }
    return densities;
}