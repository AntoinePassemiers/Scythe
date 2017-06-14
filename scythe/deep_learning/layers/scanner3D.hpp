/**
    scanner3D.hpp
    Multi-grained scanning (3D Convolutional layer)

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#ifndef SCANNER3D_HPP_
#define SCANNER3D_HPP_

#include "../../misc/sets.hpp"
#include "layer.hpp"


class ScannedDataset3D : public VirtualDataset {
private:
    size_t N;  // Number of instances
    size_t M;  // Instance height
    size_t P;  // Instance width
    size_t Q;  // Instance depth
    size_t kc; // Kernel width
    size_t kr; // Kernel height
    size_t kd; // Kernel depth
    size_t sc; // Number of kernel positions per column
    size_t sr; // Number of kernel positions per row
    size_t sd; // Number of kernel positions per depth index

    size_t Nprime; // Number of instances after scanning
    size_t Mprime; // Number of features after scanning
public:
    ScannedDataset3D(size_t kc, size_t kr, size_t kd);
    ~ScannedDataset3D() = default;
    data_t operator()(size_t i, size_t j);
    size_t getNumInstances();
    size_t getNumFeatures();
    size_t getRequiredMemorySize();
};


class MultiGrainedScanner3D : public Layer {
public:
    MultiGrainedScanner3D(LayerConfig lconfig, size_t kc, size_t kr, size_t kd);
    ~MultiGrainedScanner3D() = default;
    vdataset_p virtualize(MDDataset dataset);
    size_t getRequiredMemorySize();
};

#endif // SCANNER3D_HPP_