/**
    deep_forest.cpp
    Deep forest

    @author Antoine Passemiers
    @version 1.0 12/06/2017
*/

#include "deep_forest.hpp"


DeepForest::DeepForest(int task) : 
    layers(), 
    task(task), 
    front(nullptr), 
    rear(nullptr), 
    cascade_buffer(nullptr),
    cascade_buffer_size(0) {}

DeepForest::DeepForest(const DeepForest& other) :
    layers(other.layers),
    task(other.task),
    front(other.front),
    rear(other.rear),
    cascade_buffer(other.cascade_buffer),
    cascade_buffer_size(other.cascade_buffer_size) {}

DeepForest& DeepForest::operator=(const DeepForest& other) {
    this->layers = other.layers;
    this->task = other.task;
    this->front = other.front;
    this->rear = other.rear;
    this->cascade_buffer = other.cascade_buffer;
    this->cascade_buffer_size = other.cascade_buffer_size;
    return *this;
}

void DeepForest::add(layer_p layer) {
    layers.push_back(layer);
    if (front.get() == nullptr) {
        front = layer;
        assert(layer.get() != nullptr);
        assert(front.get() == layer.get());
    }
    if (rear.get() != nullptr) {
        rear.get()->add(layer);
    }
    rear = layer;
    assert(rear.get() == layer.get());
}

void DeepForest::add(layer_p parent, layer_p child) {
    assert(parent.get() != child.get());
    parent.get()->add(child);
    rear = child;
}

size_t DeepForest::allocateCascadeBuffer(MDDataset dataset) {
    if (cascade_buffer != nullptr) {
        delete[] cascade_buffer;
    }
    size_t required_size = 0;
    std::shared_ptr<VirtualDataset> current_vdataset;
    std::queue<layer_p> queue;
    queue.push(front);
    while (!queue.empty()) {
        layer_p current_layer = queue.front(); queue.pop();
        assert(current_layer.get() != nullptr);
        Layer* l = current_layer.get();
        current_vdataset = l->virtualize(dataset);
        if (current_layer.get()->getRequiredMemorySize() > required_size) {
            required_size = current_layer.get()->getRequiredMemorySize();
        }
        for (layer_p child : current_layer.get()->getChildren()) {
            queue.push(child);
        }
    }
    cascade_buffer = static_cast<data_t*>(malloc(required_size * sizeof(data_t)));
    return required_size;
}

void DeepForest::fit(MDDataset dataset, Labels<target_t>* labels) {
    this->cascade_buffer_size = allocateCascadeBuffer(dataset);
    vdataset_p current_vdataset;
    vtargets_p current_vtargets;
    std::queue<layer_p> queue;
    queue.push(front);
    while (!queue.empty()) {
        layer_p current_layer = queue.front(); queue.pop();
        current_vdataset = current_layer.get()->virtualize(dataset);
        current_vtargets = current_layer.get()->virtualizeTargets(labels);
        for (layer_p child : current_layer.get()->getChildren()) {
            queue.push(child);
        }
    }
}

float* DeepForest::classify(MDDataset dataset) {
    this->cascade_buffer_size = allocateCascadeBuffer(dataset);
    return nullptr; // TODO
}