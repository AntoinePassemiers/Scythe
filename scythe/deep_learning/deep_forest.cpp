/**
    deep_forest.cpp
    Deep forest

    @author Antoine Passemiers
    @version 1.0 12/06/2017
*/

#include "deep_forest.hpp"


DeepForest::DeepForest(int task) : layers(), task(task) {}

void DeepForest::add(layer_p layer) {
    layers.push_back(layer);
}

void DeepForest::fit(Dataset* dataset, Labels<target_t>* labels) {
    // TODO
}

float* DeepForest::classify(Dataset* dataset) {
    return nullptr; // TODO
}