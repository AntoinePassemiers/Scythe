/**
    deep_scythe.cpp
    Scythe's deep learning C API

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "deep_scythe.hpp"


CppClassesInterface cpp_classes_interface = CppClassesInterface();

scythe::DeepForest* CppClassesInterface::get(size_t i) {
    return df_ptrs.at(i);
}

extern "C" {

    size_t c_create_deep_forest(int task) {
        scythe::DeepForest* forest = new scythe::DeepForest(task);
        size_t ptr_id = cpp_classes_interface.num_df_ptrs++;
        cpp_classes_interface.df_ptrs.push_back(forest);
        return ptr_id;
    }

    void c_fit_deep_forest(
        scythe::MDDataset dataset, scythe::Labels* labels, size_t forest_id) {
        scythe::DeepForest* forest = cpp_classes_interface.get(forest_id);
        forest->fit(dataset, labels);
    }

    float* c_deep_forest_classify(scythe::MDDataset dataset, size_t forest_id) {
        scythe::DeepForest* forest = cpp_classes_interface.get(forest_id);
        return forest->classify(dataset);
    }

    void c_add_cascade_layer(size_t forest_id, scythe::LayerConfig lconfig) {
        scythe::DeepForest* forest = cpp_classes_interface.get(forest_id);
        scythe::layer_p layer = std::shared_ptr<scythe::CascadeLayer>(
            new scythe::CascadeLayer(lconfig));
        forest->add(layer);
    }

    void c_add_scanner_1d(size_t forest_id, scythe::LayerConfig lconfig, size_t kc) {
        scythe::DeepForest* forest = cpp_classes_interface.get(forest_id);
        scythe::layer_p layer = std::shared_ptr<scythe::MultiGrainedScanner1D>(
            new scythe::MultiGrainedScanner1D(lconfig, kc));
        forest->add(layer);
    }

    void c_add_scanner_2d(size_t forest_id, scythe::LayerConfig lconfig, size_t kc, size_t kr) {
        scythe::DeepForest* forest = cpp_classes_interface.get(forest_id);
        scythe::layer_p layer = std::shared_ptr<scythe::MultiGrainedScanner2D>(
            new scythe::MultiGrainedScanner2D(lconfig, kc, kr));
        forest->add(layer);
    }

    void c_add_scanner_3d(size_t forest_id, scythe::LayerConfig lconfig, size_t kc, size_t kr, size_t kd) {
        scythe::DeepForest* forest = cpp_classes_interface.get(forest_id);
        scythe::layer_p layer = std::shared_ptr<scythe::MultiGrainedScanner3D>(
            new scythe::MultiGrainedScanner3D(lconfig, kc, kr, kd));
        forest->add(layer);
    }
}