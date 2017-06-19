/**
    layer.hpp
    Deep learning base layer

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "layer.hpp"


LayerConfig::LayerConfig() :
     fconfig(), n_forests(0), forest_type(gbdf::COMPLETE_RANDOM_FOREST) {}

Layer::Layer(LayerConfig lconfig) :
    name(), in_shape(), virtual_in_shape(), virtual_out_shape(), 
    children(), forests(), vdataset(nullptr), vtargets(nullptr), lconfig() {
    this->lconfig = lconfig;

    ForestConfig* fconfig = &(lconfig.fconfig);
    std::shared_ptr<Forest> new_forest;
    for (size_t i = 0; i < lconfig.n_forests; i++) {
        if (lconfig.forest_type == gbdf::RANDOM_FOREST) {
            /**
            new_forest = std::shared_ptr(
                new )
            */
        }
    }
}

void Layer::add(layer_p layer) {
    children.push_back(layer);
}

vdataset_p Layer::getVirtualDataset() {
    return this->vdataset;
}

size_t Layer::getNumChildren() {
    return this->children.size();
}