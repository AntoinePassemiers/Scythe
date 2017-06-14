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

    void c_fit_deep_forest(MDDataset dataset, Labels<target_t>* labels, void* forest_p) {
        DeepForest* forest = static_cast<DeepForest*>(forest_p);
        forest->fit(dataset, labels);
    }

    float* c_deep_forest_classify(MDDataset dataset, void* forest_p) {
        DeepForest* forest = static_cast<DeepForest*>(forest_p);
        return forest->classify(dataset);
    }

    void c_add_scanner_1d(void* forest_p, LayerConfig lconfig, size_t kc) {
        DeepForest* forest = static_cast<DeepForest*>(forest_p);
        layer_p layer = std::shared_ptr<MultiGrainedScanner1D>(
            new MultiGrainedScanner1D(lconfig, kc));
        forest->add(layer);
    }

    void c_add_scanner_2d(void* forest_p, LayerConfig lconfig, size_t kc, size_t kr) {
        DeepForest* forest = static_cast<DeepForest*>(forest_p);
        layer_p layer = std::shared_ptr<MultiGrainedScanner2D>(
            new MultiGrainedScanner2D(lconfig, kc, kr));
        forest->add(layer);
    }

    void c_add_scanner_3d(void* forest_p, LayerConfig lconfig, size_t kc, size_t kr, size_t kd) {
        DeepForest* forest = static_cast<DeepForest*>(forest_p);
        layer_p layer = std::shared_ptr<MultiGrainedScanner3D>(
            new MultiGrainedScanner3D(lconfig, kc, kr, kd));
        forest->add(layer);
    }

}