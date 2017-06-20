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
    if (front.get() == nullptr) {
        front = layer;
    }
    if (rear.get() != nullptr) {
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
        std::cout << "A" << std::endl;
        assert(current_layer.get() != nullptr);
        std::cout << "C" << std::endl;
        Layer* l = current_layer.get();
        std::cout << "D" << std::endl;
        std::cout << l << std::endl;
        current_vdataset = l->virtualize(dataset);
        std::cout << "B" << std::endl;
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
    vdataset_p current_vdataset;
    vtargets_p current_vtargets;
    std::queue<layer_p> queue;
    queue.push(front);
    while (!queue.empty()) {
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