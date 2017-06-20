/**
    scanner2D.hpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#ifndef SCANNER2D_HPP_
#define SCANNER2D_HPP_

#include "../../misc/sets.hpp"
#include "layer.hpp"


class ScannedDataset2D : public VirtualDataset {
private:
    size_t N;  // Number of instances
    size_t M;  // Instance height
    size_t P;  // Instance width
    size_t kc; // Kernel width
    size_t kr; // Kernel height
    size_t sc; // Number of kernel positions per column
    size_t sr; // Number of kernel positions per row

    size_t Nprime; // Number of instances after scanning
    size_t Mprime; // Number of features after scanning

    data_t* data; // Pointer to the raw data
public:
    ScannedDataset2D(data_t* data, size_t N, size_t M, size_t P, size_t kc, size_t kr);
    ~ScannedDataset2D() = default;
    data_t operator()(size_t i, size_t j);
    size_t getSc();
    size_t getSr();
    size_t getNumInstances();
    size_t getNumFeatures();
    size_t getRequiredMemorySize();
};


class ScannedTargets2D : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;
    size_t s;
public:
    ScannedTargets2D(data_t* data, size_t n_instances, size_t sc, size_t sr);
    ScannedTargets2D(const ScannedTargets2D& other);
    ScannedTargets2D& operator=(const ScannedTargets2D& other);
    ~ScannedTargets2D() = default;
    data_t operator[](const size_t i);
    size_t getNumInstances() { return n_rows; }
    target_t* getValues() { return data; }
};


class MultiGrainedScanner2D : public Layer {
private:
    size_t kc; // Kernel width
    size_t kr; // Kernel height
public:
    MultiGrainedScanner2D(LayerConfig lconfig, size_t kc, size_t kr);
    ~MultiGrainedScanner2D() = default;
    vdataset_p virtualize(MDDataset dataset);
    vtargets_p virtualize(Labels<target_t>* targets);
    size_t getRequiredMemorySize();
};

#endif // SCANNER2D_HPP_