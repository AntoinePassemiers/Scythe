/**
    deep_scythe.hpp
    Scythe's deep learning C API

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#ifndef DEEP_SCYTHE_HPP_
#define DEEP_SCYTHE_HPP_

#include "deep_learning/deep_forest.hpp"
#include "deep_learning/layers/layer.hpp"
#include "deep_learning/layers/concatenation_layer.hpp"
#include "deep_learning/layers/scanner1D.hpp"
#include "deep_learning/layers/scanner2D.hpp"
#include "deep_learning/layers/scanner3D.hpp"


struct CppClassesInterface {
    size_t num_df_ptrs;
    std::vector<scythe::DeepForest*> df_ptrs;
    CppClassesInterface() : num_df_ptrs(0), df_ptrs() {}
    scythe::DeepForest* get(size_t i);
};

extern CppClassesInterface cpp_classes_interface;

extern "C" {

    size_t c_create_deep_forest(int task);

    void c_fit_deep_forest(scythe::MDDataset dataset, scythe::Labels* labels, size_t forest_id);

    float* c_deep_forest_classify(scythe::MDDataset dataset, size_t forest_id);

    void c_add_cascade_layer(size_t forest_id, scythe::LayerConfig lconfig);

    void c_add_scanner_1d(size_t forest_id, scythe::LayerConfig lconfig, size_t kc);

    void c_add_scanner_2d(size_t forest_id, scythe::LayerConfig lconfig, size_t kc, size_t kr);

    void c_add_scanner_3d(size_t forest_id, scythe::LayerConfig lconfig, size_t kc, size_t kr, size_t kd);

    /**
    void c_add_direct_layer(void* forest_p, LayerConfig lconfig);
    */

}

#endif // DEEP_SCYTHE_HPP_