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
        current_vdataset = current_layer.get()->virtualize(dataset);
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
    layer_p current_layer = front;
    std::shared_ptr<VirtualDataset> current_vdataset;
    do {
        current_vdataset = current_layer.get()->virtualize(dataset);
    } while (current_layer.get()->getNumChildren() > 0);
}

float* DeepForest::classify(MDDataset dataset) {
    this->cascade_buffer_size = allocateCascadeBuffer(dataset);
    return nullptr; // TODO
}