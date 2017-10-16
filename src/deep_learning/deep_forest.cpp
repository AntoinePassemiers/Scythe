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
    cascade_buffers() {}

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

void DeepForest::connect(layer_p parent, layer_p child) {
    assert(parent.get() != child.get());
    rear = child;
    // TODO: assert child and parent in this.layers
    parent->addChild(child);
    child->addParent(parent);
}

void DeepForest::printGraph() {
    size_t layer_id = 1;
    std::cout << std::endl;
    for (layer_p layer : layers) {
        std::cout << "----------------------" << std::endl;
        std::cout << "Layer " << std::setfill(' ') << std::setw(3) << layer_id << std::endl;
        std::cout << "        | Type       : " << layer->getType() << std::endl;
        std::cout << "        | n_parents  : " << layer->getNumParents() << std::endl;
        std::cout << "        | n_children : " << layer->getNumChildren() << std::endl;
        layer_id++;
    }
    std::cout << "----------------------\n" << std::endl;
}

size_t DeepForest::allocateCascadeBuffer(MDDataset dataset) {
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

    cascade_buffers = std::queue<std::shared_ptr<ConcatenationDataset>>();
    for (size_t i = 0; i < 2; i++) {
        std::shared_ptr<ConcatenationDataset> cascade_buffer = std::shared_ptr<ConcatenationDataset>(
            new ConcatenationDataset(n_instances, n_virtual_cols));
        cascade_buffers.push(cascade_buffer);
    }
    return n_instances * n_virtual_cols;
}

void DeepForest::transfer(layer_p layer,
    vdataset_p vdataset, std::shared_ptr<ConcatenationDataset> buffer) {
    // buffer->reset();
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
    printGraph();

    size_t n_instances = dataset.dims[0];
    std::shared_ptr<VirtualTargets> direct_targets = std::shared_ptr<VirtualTargets>(
        new DirectTargets(labels->data, n_instances));
    allocateCascadeBuffer(dataset);
    std::queue<layer_p> queue;
    for (layer_p layer : layers) {
        if (!layer->isConcatenable()) {
            vdataset_p current_vdataset = layer->virtualize(dataset);
            vtargets_p current_vtargets = layer->virtualizeTargets(labels);

            vdataset_p copied_current_vdataset = std::shared_ptr<VirtualDataset>(current_vdataset->deepcopy());
            vtargets_p copied_current_vtargets = std::shared_ptr<VirtualTargets>(current_vtargets->deepcopy());
            layer->grow(copied_current_vdataset, copied_current_vtargets);

            for (layer_p child : layer->getChildren()) {
                queue.push(child);
            }
        }
    }
    while (!queue.empty()) {
        cascade_buffers.front()->reset();
        cascade_buffers.back()->reset();

        layer_p layer = queue.front(); queue.pop();
        for (layer_p parent : layer->getParents()) {
            if (!parent->isConcatenable()) {
                vdataset_p current_vdataset = parent->virtualize(dataset);
                transfer(parent, current_vdataset, cascade_buffers.front());
            }
            else {
                std::shared_ptr<ConcatenationDataset> cb = cascade_buffers.front(); cascade_buffers.pop();
                cascade_buffers.push(cb);
                transfer(parent, cb, cascade_buffers.front());
            }
        }

        vdataset_p copied_front_buffer = std::shared_ptr<VirtualDataset>(cascade_buffers.front()->deepcopy());
        vtargets_p copied_direct_targets = std::shared_ptr<VirtualTargets>(direct_targets->deepcopy());
        layer->grow(copied_front_buffer, copied_direct_targets);
        for (layer_p child : layer->getChildren()) {
            queue.push(child);
        }
    }
}

float* DeepForest::classify(MDDataset dataset) {
    allocateCascadeBuffer(dataset);
    std::queue<layer_p> queue;
    for (layer_p layer : layers) {
        if (!layer->isConcatenable()) {
            for (layer_p child : layer->getChildren()) {
                queue.push(child);
            }
        }
    }
    layer_p layer;
    while (!queue.empty()) {
        cascade_buffers.front()->reset();
        cascade_buffers.back()->reset();

        layer = queue.front(); queue.pop();
        for (layer_p parent : layer->getParents()) {
            if (!parent->isConcatenable()) {
                vdataset_p current_vdataset = parent->virtualize(dataset);
                transfer(parent, current_vdataset, cascade_buffers.front());
            }
            else {
                std::shared_ptr<ConcatenationDataset> cb = cascade_buffers.front(); cascade_buffers.pop();
                cascade_buffers.push(cb);
                transfer(parent, cb, cascade_buffers.front());
            }
        }
        for (layer_p child : layer->getChildren()) {
            queue.push(child);
        }
    }
    return layer->classify(cascade_buffers.front());
}

} // namespace