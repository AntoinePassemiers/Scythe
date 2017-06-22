/**
    deep_scythe.cpp
    Scythe's deep learning C API

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "deep_scythe.hpp"


CppClassesInterface cpp_classes_interface = CppClassesInterface();

DeepForest* CppClassesInterface::get(size_t i) {
    return df_ptrs.at(i);
}

extern "C" {

    size_t c_create_deep_forest(int task) {
        DeepForest* forest = new DeepForest(task);
        size_t ptr_id = cpp_classes_interface.num_df_ptrs++;
        cpp_classes_interface.df_ptrs.push_back(forest);
        return ptr_id;
    }

    void c_fit_deep_forest(MDDataset dataset, Labels<target_t>* labels, size_t forest_id) {
        DeepForest* forest = cpp_classes_interface.get(forest_id);
        forest->fit(dataset, labels);
    }

    float* c_deep_forest_classify(MDDataset dataset, void* forest_p) {
        DeepForest* forest = static_cast<DeepForest*>(forest_p);
        return forest->classify(dataset);
    }

    void c_add_scanner_1d(size_t forest_id, LayerConfig lconfig, size_t kc) {
        DeepForest* forest = cpp_classes_interface.get(forest_id);
        layer_p layer = std::shared_ptr<MultiGrainedScanner1D>(
            new MultiGrainedScanner1D(lconfig, kc));
        forest->add(layer);
    }

    void c_add_scanner_2d(size_t forest_id, LayerConfig lconfig, size_t kc, size_t kr) {
        DeepForest* forest = cpp_classes_interface.get(forest_id);
        layer_p layer = std::shared_ptr<MultiGrainedScanner2D>(
            new MultiGrainedScanner2D(lconfig, kc, kr));
        forest->add(layer);
    }

    void c_add_scanner_3d(size_t forest_id, LayerConfig lconfig, size_t kc, size_t kr, size_t kd) {
        DeepForest* forest = cpp_classes_interface.get(forest_id);
        layer_p layer = std::shared_ptr<MultiGrainedScanner3D>(
            new MultiGrainedScanner3D(lconfig, kc, kr, kd));
        forest->add(layer);
    }
}