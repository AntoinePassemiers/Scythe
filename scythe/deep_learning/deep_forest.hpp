/**
    deep_forest.hpp
    Deep forest

    @author Antoine Passemiers
    @version 1.0 12/06/2017
*/

#ifndef DEEP_FOREST_HPP_
#define DEEP_FOREST_HPP_

#include "layers/layer.hpp"


class DeepForest {
private:
    std::vector<layer_p> layers;
    int task;
    layer_p front;
    layer_p rear;
    data_t* cascade_buffer;
public:
    DeepForest(int task);
    ~DeepForest() = default;
    void fit(MDDataset dataset, Labels<target_t>* labels);
    float* classify(MDDataset dataset);
    void add(layer_p layer);
    void add(layer_p parent, layer_p child);
    void allocateCascadeBuffer(MDDataset dataset);
};

#endif // DEEP_FOREST_HPP_