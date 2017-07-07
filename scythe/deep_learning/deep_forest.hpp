/**
    deep_forest.hpp
    Deep forest

    @author Antoine Passemiers
    @version 1.0 12/06/2017
*/

#ifndef DEEP_FOREST_HPP_
#define DEEP_FOREST_HPP_

#include <algorithm>
#include <queue>

#include "layers/layer.hpp"
#include "layers/concatenation_layer.hpp"


class DeepForest {
private:
    std::vector<layer_p> layers;
    int task;
    layer_p front;
    layer_p rear;
    std::shared_ptr<ConcatenationDataset> cascade_buffer;
public:
    DeepForest(int task);
    DeepForest(const DeepForest& other) = default;
    DeepForest& operator=(const DeepForest& other) = default;
    ~DeepForest() = default;
    void fit(MDDataset dataset, Labels<target_t>* labels);
    float* classify(MDDataset dataset);
    void add(layer_p layer);
    void add(layer_p parent, layer_p child);
    size_t allocateCascadeBuffer(MDDataset dataset);
    void transfer(layer_p, vdataset_p, std::shared_ptr<ConcatenationDataset>);
    layer_p getFront() { return front; }
};

#endif // DEEP_FOREST_HPP_