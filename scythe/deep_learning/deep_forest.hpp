/**
    deep_forest.hpp
    Deep forest

    @author Antoine Passemiers
    @version 1.0 12/06/2017
*/

#ifndef DEEP_FOREST_HPP_
#define DEEP_FOREST_HPP_

#include "layers/layer.hpp"

typedef std::shared_ptr<Layer> layer_p;


class DeepForest {
private:
    std::vector<layer_p> layers;
    int task;
public:
    DeepForest(int task);
    ~DeepForest() = default;
    void add(layer_p layer);
};

#endif // DEEP_FOREST_HPP_