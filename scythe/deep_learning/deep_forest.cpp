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
    cascade_buffer(nullptr) {}

void DeepForest::add(layer_p layer) {
    layer->setTask(task);
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
    size_t n_virtual_features = 0;
    size_t required_size = 0;
    std::shared_ptr<VirtualDataset> current_vdataset;
    std::queue<layer_p> queue;
    queue.push(front);
    while (!queue.empty()) {
        layer_p current_layer = queue.front(); queue.pop();
        assert(current_layer.get() != nullptr);
        if (current_layer->getChildren().size() > 0) {
            current_vdataset = current_layer->virtualize(dataset);
            required_size += current_layer->getRequiredMemorySize();
            n_virtual_features += current_layer->getNumVirtualFeatures();
            for (layer_p child : current_layer->getChildren()) {
                queue.push(child);
            }
        }
    }
    assert(required_size == dataset.dims[0] * n_virtual_features);
    cascade_buffer = std::shared_ptr<ConcatenationDataset>(
        new ConcatenationDataset(dataset.dims[0], n_virtual_features));
    return required_size;
}

void DeepForest::transfer(layer_p layer, vdataset_p vdataset, std::shared_ptr<ConcatenationDataset> buffer) {
    for (std::shared_ptr<Forest> forest_p : layer->getForests()) {
        if (layer->isClassifier()) {
            float* predictions = dynamic_cast<ClassificationForest*>(
                forest_p.get())->classify(vdataset.get());
            size_t vipi = vdataset->getNumVirtualInstancesPerInstance();
            size_t stride = vipi * forest_p->getInstanceStride();
            buffer->concatenate(predictions, stride);
            delete[] predictions;
        }
        // TODO : regression
    }
}

void DeepForest::fit(MDDataset dataset, Labels<target_t>* labels) {
    size_t n_instances = dataset.dims[0];
    DirectTargets* direct_targets = new DirectTargets(
        labels->data, n_instances);
    std::cout << "A" << std::endl;
    allocateCascadeBuffer(dataset);
    std::cout << "B" << std::endl;
    vdataset_p current_vdataset;
    vtargets_p current_vtargets;
    std::queue<layer_p> queue;
    queue.push(front);
    std::cout << "C" << std::endl;
    current_vdataset = front->virtualize(dataset);
    current_vtargets = front->virtualizeTargets(labels);
    std::cout << "D" << std::endl;
    front->grow(current_vdataset, current_vtargets);
    std::cout << "E" << std::endl;
    while (!queue.empty()) {
        std::cout << "F" << std::endl;
        layer_p current_layer = queue.front(); queue.pop();
        if (current_layer->getChildren().size() > 0) {
            transfer(current_layer, current_vdataset, cascade_buffer);
            current_vdataset = cascade_buffer;
            std::cout << "G" << std::endl;
            current_vtargets = std::shared_ptr<VirtualTargets>(direct_targets);
            for (layer_p child : current_layer->getChildren()) {
                std::cout << "H" << std::endl;
                current_vtargets = child->virtualizeTargets(labels);
                std::cout << "I" << std::endl;
                child->grow(cascade_buffer, current_vtargets);
                queue.push(child);
                std::cout << "J" << std::endl;
            }
        }
        std::cout << "K" << std::endl;
    }
    std::cout << "L" << std::endl;
}

float* DeepForest::classify(MDDataset dataset) {
    allocateCascadeBuffer(dataset);
    std::queue<layer_p> queue;
    queue.push(front);
    layer_p current_layer;
    vdataset_p current_vdataset;
    while (!queue.empty()) {
        current_vdataset = front->virtualize(dataset);
        current_layer = queue.front(); queue.pop();
        if (current_layer->getChildren().size() > 0) {
            transfer(current_layer, current_vdataset, cascade_buffer);
            for (layer_p child : current_layer->getChildren()) {
                queue.push(child);
            }
        }
        else {
            break;
        }
        current_vdataset = cascade_buffer;
    }
    return current_layer->classify(current_vdataset);
}