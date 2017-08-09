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


namespace scythe {

class DeepForest {
private:
    std::vector<layer_p> layers;
    size_t n_layers;
    int task;
    layer_p front;
    layer_p rear;
    std::shared_ptr<ConcatenationDataset> cascade_buffer;
public:
    DeepForest(int task);
    DeepForest(const DeepForest& other) = default;
    DeepForest& operator=(const DeepForest& other) = default;
    ~DeepForest() = default;
    void fit(MDDataset dataset, Labels* labels);
    float* classify(MDDataset dataset);
    size_t add(layer_p layer);
    size_t add(layer_p parent, layer_p child);
    std::vector<layer_p> iterateOverLayers();
    size_t allocateCascadeBuffer(MDDataset dataset);
    void transfer(layer_p, vdataset_p, std::shared_ptr<ConcatenationDataset>);

    layer_p getFront() { return front; }
    layer_p getLayerByID(size_t layer_id) { return layers.at(layer_id); }
};

} // namespace

#endif // DEEP_FOREST_HPP_