/**
    scanner1D.hpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#ifndef SCANNER1D_HPP_
#define SCANNER1D_HPP_

#include "../../misc/sets.hpp"
#include "layer.hpp"

namespace scythe {

class ScannedDataset1D : public VirtualDataset {
private:
    size_t N;  // Number of instances
    size_t M;  // Number of features
    size_t kc; // Kernel width
    size_t sc; // Number of kernel positions per column

    size_t Nprime; // Number of instances after scanning
    size_t Mprime; // Number of features after scanning

    data_t* data; // Pointer to the raw data
    int dtype;    // Raw data type
public:
    ScannedDataset1D(data_t* data, size_t N, size_t M, size_t kc, int dtype);
    ScannedDataset1D(const ScannedDataset1D& other) = default;
    ScannedDataset1D& operator=(const ScannedDataset1D& other) = default;
    ~ScannedDataset1D() override = default;
    size_t getSc() { return sc; }
    virtual data_t operator()(size_t i, size_t j);
    virtual std::shared_ptr<void> _operator_ev(const size_t j); // Type erasure
    virtual size_t getNumInstances() { return Nprime; }
    virtual size_t getNumFeatures() { return Mprime; }
    virtual size_t getRequiredMemorySize() { return Nprime * Mprime; }
    virtual size_t getNumVirtualInstancesPerInstance() { return sc; }
    virtual int getDataType() { return dtype; }
};


class ScannedTargets1D : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;
    size_t s;
public:
    ScannedTargets1D(target_t* data, size_t n_instances, size_t sc);
    ScannedTargets1D(const ScannedTargets1D& other) = default;
    ScannedTargets1D& operator=(const ScannedTargets1D& other) = default;
    ~ScannedTargets1D() override = default;
    virtual target_t operator[](const size_t i);
    virtual size_t getNumInstances() { return n_rows; }
    virtual target_t* getValues() { return data; }
};


class MultiGrainedScanner1D : public Layer {
private:
    size_t kc; // Kernel width
public:
    MultiGrainedScanner1D(LayerConfig lconfig, size_t kc);
    ~MultiGrainedScanner1D() {}
    virtual vdataset_p virtualize(MDDataset dataset);
    virtual vtargets_p virtualizeTargets(Labels* targets);
    virtual size_t getRequiredMemorySize();
    virtual size_t getNumVirtualFeatures();
    virtual bool isConcatenable() { return false; }
    virtual std::string getType() { return std::string("MultiGrainedScanner1D"); }
};

}

#endif // SCANNER1D_HPP_