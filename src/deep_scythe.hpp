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
#include "deep_learning/layers/scanner2D.hpp"


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

    size_t c_add_cascade_layer(size_t forest_id, scythe::LayerConfig lconfig);

    size_t c_add_scanner_2d(size_t forest_id, scythe::LayerConfig lconfig, size_t kc, size_t kr);

    void c_connect_nodes(size_t forest_id, size_t parent_id, size_t child_id);

    void* c_get_forest(size_t deep_forest_id, size_t layer_id, size_t forest_id);

}

#endif // DEEP_SCYTHE_HPP_