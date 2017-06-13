/**
    layer.hpp
    Deep learning base layer

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "layer.hpp"


Layer::Layer(LayerConfig lconfig) {
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