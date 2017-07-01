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

    data_t* data; // Pointer to the raw data
    int dtype;    // Raw data type
public:
    ScannedDataset3D(data_t* data, size_t kc, size_t kr, size_t kd, int dtype);
    ScannedDataset3D(const ScannedDataset3D& other) = default;
    ScannedDataset3D& operator=(const ScannedDataset3D& other) = default;
    ~ScannedDataset3D() override = default;
    size_t getSc();
    size_t getSr();
    size_t getSd();
    virtual data_t operator()(size_t i, size_t j);
    virtual std::shared_ptr<void> _operator_ev(const size_t j); // Type erasure
    virtual size_t getNumInstances();
    virtual size_t getNumFeatures();
    virtual size_t getRequiredMemorySize();
    virtual size_t getNumVirtualInstancesPerInstance();
    virtual int getDataType() { return dtype; }
};


class ScannedTargets3D : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;
    size_t s;
public:
    ScannedTargets3D(data_t* data, size_t n_instances, size_t sc, size_t sr, size_t sd);
    ScannedTargets3D(const ScannedTargets3D& other);
    ScannedTargets3D& operator=(const ScannedTargets3D& other);
    ~ScannedTargets3D() override = default;
    virtual data_t operator[](const size_t i);
    virtual size_t getNumInstances() { return n_rows; }
    virtual target_t* getValues() { return data; }
};


class MultiGrainedScanner3D : public Layer {
private:
    size_t kc; // Kernel width
    size_t kr; // Kernel height
    size_t kd; // Kernel depth
public:
    MultiGrainedScanner3D(LayerConfig lconfig, size_t kc, size_t kr, size_t kd);
    ~MultiGrainedScanner3D() = default;
    virtual vdataset_p virtualize(MDDataset dataset);
    virtual vtargets_p virtualizeTargets(Labels<target_t>* targets);
    virtual size_t getRequiredMemorySize();
    virtual size_t getNumVirtualFeatures();
    virtual bool isConcatenable() { return false; }
    virtual std::string getType() { return std::string("MultiGrainedScanner3D"); }
};

#endif // SCANNER3D_HPP_