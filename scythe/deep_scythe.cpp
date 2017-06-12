/**
    deep_scythe.cpp
    Scythe's deep learning C API

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "deep_scythe.hpp"


extern "C" {

    void* c_create_deep_forest(int task) {
        DeepForest* forest = new DeepForest(task);
        return static_cast<void*>(forest);
    }

    void c_add_scanner_1d(void* forest_p, LayerConfig lconfig, size_t kc) {
        DeepForest* forest = static_cast<DeepForest*>(forest_p);
        layer_p layer = std::shared_ptr<MultiGrainedScanner1D>(
            new MultiGrainedScanner1D(lconfig, kc));
        forest->add(layer);
    }

}