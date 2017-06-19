/**
    layer.hpp
    Deep learning base layer

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <memory>
#include <limits>
#include <cassert>
#include <string.h>

#include "../../forest/forest.hpp"
#include "../../forest/classification_complete_rf.hpp"
#include "../../forest/classification_rf.hpp"
#include "../../forest/classification_gb.hpp"
// #include "../../forest/regression_complete_rf.hpp"
// #include "../../forest/regression_rf.hpp"
// #include "../../forest/regression_gb.hpp"
    

class Layer; // Forward declaration

typedef std::shared_ptr<Layer> layer_p;

struct LayerConfig {
    ForestConfig fconfig;
    size_t       n_forests;
    int          forest_type;
    LayerConfig();
};

/**
    Main goal of layers: ensuring that each forest gets
    a two-dimensional dataset as input, and ensuring that
    the dimensionality of the output is right
    (1d for regression, 2d for classification). This dimensionality
    must be invariant to the complexity of cascades and convolutional layers.

    Therefore, the shapes of the datasets are "re-mapped" between layers, and
    the definition of how it works must be defined in each layer class.
*/

class Layer {
protected:
    std::string name; // Layer name

    // The product of in_shape elements must be equal to the product of
    // virtual_in_shape elements.
    std::vector<size_t> in_shape;          // Input shape
    std::vector<size_t> virtual_in_shape;  // Virtual input shape
    std::vector<size_t> virtual_out_shape; // Virtual output shape

    std::vector<layer_p> children; // children layers
    std::vector<std::shared_ptr<Forest>> forests; // Intern forests

    vdataset_p vdataset; // Virtual dataset
    vtargets_p vtargets; // Virtual target values
    LayerConfig lconfig; // Layer configuration

    bool grown = false; // Indicates whether the layer has learned or not

public:
    Layer(LayerConfig lconfig);
    ~Layer() = default;
    void add(layer_p layer);
    virtual vdataset_p virtualize(MDDataset dataset) = 0;
    size_t getNumChildren();
    std::vector<layer_p> getChildren() { return children; }
    vdataset_p getVirtualDataset();
    virtual size_t getRequiredMemorySize() = 0;
    void grow(VirtualDataset* vdataset, VirtualTargets* vtargets);
};

#endif // LAYER_HPP_