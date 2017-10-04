/**
    deep_forest.cpp
    Deep forest

    @author Antoine Passemiers
    @version 1.0 12/06/2017
*/

#include "deep_forest.hpp"


namespace scythe {

DeepForest::DeepForest(int task) : 
    layers(),
    n_layers(0),
    task(task), 
    front(nullptr), 
    rear(nullptr), 
    cascade_buffer(nullptr) {}

size_t DeepForest::add(layer_p layer) {
    layer->setTask(task);
    layers.push_back(layer);
    if (front.get() == nullptr) {
        front = layer;
        assert(layer.get() != nullptr);
        assert(front.get() == layer.get());
    }
    if (rear.get() != nullptr) {
        rear->addChild(layer);
        layer->addParent(rear);
    }
    rear = layer;
    assert(rear.get() == layer.get());
    return n_layers++;
}

size_t DeepForest::add(layer_p parent, layer_p child) {
    assert(parent.get() != child.get());
    rear = child;
    parent->addChild(child);
    child->addParent(parent);
    child->setTask(task);
    layers.push_back(child);
    return n_layers++;
}

size_t DeepForest::allocateCascadeBuffer(MDDataset dataset) {
    /**
    size_t required_mgs_size = 0;
    size_t required_cascade_size = 0;
    for (layer_p current_layer : layers) {
        assert(current_layer.get() != nullptr);
        std::shared_ptr<VirtualDataset> current_vdataset = current_layer->virtualize(dataset);
        size_t n_forests = current_layer->getNumForests();
        if (!current_layer->isConcatenable()) {
            required_mgs_size = std::max(
                required_mgs_size,
                current_layer->getNumVirtualFeatures());
        }
        else if (current_layer->getChildren().size() > 0) {
            required_cascade_size = std::max(
                required_cascade_size,
                current_layer->getNumVirtualFeatures());
        }
    }
    size_t n_instances = dataset.dims[0];
    size_t n_virtual_cols = required_mgs_size + required_cascade_size;
    this->cascade_buffer = std::shared_ptr<ConcatenationDataset>(
        new ConcatenationDataset(n_instances, n_virtual_cols));
    return n_instances * n_virtual_cols;
    */
    size_t required_mgs_size = 0;
    size_t required_cascade_size = 0;
    for (layer_p current_layer : layers) {
        assert(current_layer.get() != nullptr);
        size_t current_layer_required_mgs_size = 0;
        for (layer_p parent : current_layer->getParents()) {
            parent->virtualize(dataset);
            current_layer_required_mgs_size += parent->getNumVirtualFeatures();
        }
        required_mgs_size = std::max(
            required_mgs_size, current_layer_required_mgs_size);
    }
    size_t n_instances = dataset.dims[0];
    size_t n_virtual_cols = required_mgs_size + required_cascade_size;

    this->cascade_buffer = std::shared_ptr<ConcatenationDataset>(
        new ConcatenationDataset(n_instances, n_virtual_cols));
    return n_instances * n_virtual_cols;
}

void DeepForest::transfer(layer_p layer,
    vdataset_p vdataset, std::shared_ptr<ConcatenationDataset> buffer) {
    buffer->reset();
    for (std::shared_ptr<Forest> forest_p : layer->getForests()) {
        if (layer->isClassifier()) {
            float* predictions = dynamic_cast<ClassificationForest*>(
                forest_p.get())->classify(vdataset.get());
            size_t vipi = vdataset->getNumVirtualInstancesPerInstance();
            size_t stride = vipi * forest_p->getInstanceStride();
            buffer->concatenate(predictions, stride);
            delete[] predictions;
        }
        else {
            assert(false); // TODO : regression
        }
    }
}

void DeepForest::fit(MDDataset dataset, Labels* labels) {
    size_t n_instances = dataset.dims[0];
    std::shared_ptr<VirtualTargets> direct_targets = std::shared_ptr<VirtualTargets>(
        new DirectTargets(labels->data, n_instances));
    allocateCascadeBuffer(dataset);
    std::queue<layer_p> queue;
    queue.push(front);
    vdataset_p current_vdataset = front->virtualize(dataset);
    vtargets_p current_vtargets = front->virtualizeTargets(labels);
    front->grow(current_vdataset, current_vtargets);
    std::cout << "AAAAAA" << std::endl;
    while (!queue.empty()) {
        layer_p current_layer = queue.front(); queue.pop();
        if (current_layer->getChildren().size() > 0) {
            std::cout << "VVVVVV" << std::endl;
            transfer(current_layer, current_vdataset, cascade_buffer);
            std::cout << "XXXXXX" << std::endl;
            current_vtargets = direct_targets;
            for (layer_p child : current_layer->getChildren()) {
                // current_vtargets = child->virtualizeTargets(labels);
                std::cout << "LLLLLL" << std::endl;
                child->grow(cascade_buffer, current_vtargets);
                std::cout << "EEEEEE" << std::endl;
                queue.push(child);
            }
            std::cout << "SSSSSS" << std::endl;
        }
        std::cout << "OOOOOO" << std::endl;
    }
    std::cout << "RRRRRR" << std::endl;
}

float* DeepForest::classify(MDDataset dataset) {
    std::cout << "QQQQQQ" << std::endl;
    allocateCascadeBuffer(dataset);
    std::cout << "KKKKKK" << std::endl;
    std::queue<layer_p> queue;
    queue.push(front);
    layer_p current_layer;
    vdataset_p current_vdataset = front->virtualize(dataset);
    std::cout << "CCCCCC" << std::endl;
    while (!queue.empty()) {
        std::cout << "BBBBBB" << std::endl;
        current_layer = queue.front(); queue.pop();
        if (current_layer->getChildren().size() > 0) {
            std::cout << "NNNNNN" << std::endl;
            transfer(current_layer, current_vdataset, cascade_buffer);
            std::cout << "FFFFFF" << std::endl;
            for (layer_p child : current_layer->getChildren()) {
                queue.push(child);
            }
        }
    }
    std::cout << "ZZZZZZ" << std::endl;
    return current_layer->classify(cascade_buffer);
}

} // namespace