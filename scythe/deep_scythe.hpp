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
#include "deep_learning/layers/scanner1D.hpp"
#include "deep_learning/layers/scanner2D.hpp"
#include "deep_learning/layers/scanner3D.hpp"


extern "C" {

    void* c_create_deep_forest(int task);

    void c_fit_deep_forest(MDDataset dataset, Labels<target_t>* labels, void* forest_p);

    float* c_deep_forest_classify(MDDataset dataset, void* forest_p);

    void c_add_scanner_1d(void* forest_p, LayerConfig lconfig, size_t kc);

    void c_add_scanner_2d(void* forest_p, LayerConfig lconfig, size_t kc, size_t kr);

    void c_add_scanner_3d(void* forest_p, LayerConfig lconfig, size_t kc, size_t kr, size_t kd);

    /**
    void c_add_direct_layer(void* forest_p, LayerConfig lconfig);

    void c_add_cascade_layer(void* forest_p, LayerConfig lconfig);
    */

}

#endif // DEEP_SCYTHE_HPP_