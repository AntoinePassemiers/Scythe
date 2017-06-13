/**
    deep_forest.cpp
    Deep forest

    @author Antoine Passemiers
    @version 1.0 12/06/2017
*/

#include "deep_forest.hpp"


DeepForest::DeepForest(int task) : 
    layers(), task(task), front(nullptr), rear(nullptr) {}

void DeepForest::add(layer_p layer) {
    layers.push_back(layer);
    if (front == nullptr) {
        front = layer;
    }
    if (rear != nullptr) {
        rear.get()->add(layer);
    }
    rear = layer;
}

void DeepForest::add(layer_p parent, layer_p child) {
    assert(parent != child);
    parent.get()->add(child);
    rear = child;
}

void DeepForest::fit(Dataset* dataset, Labels<target_t>* labels) {
    layer_p current_layer = front;
    // TODO
}

float* DeepForest::classify(Dataset* dataset) {
    return nullptr; // TODO
}