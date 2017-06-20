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


class ScannedDataset1D : public VirtualDataset {
private:
    size_t N;  // Number of instances
    size_t M;  // Number of features
    size_t kc; // Kernel width
    size_t sc; // Number of kernel positions per column

    size_t Nprime; // Number of instances after scanning
    size_t Mprime; // Number of features after scanning

    data_t* data; // Pointer to the raw data
public:
    ScannedDataset1D(data_t* data, size_t N, size_t M, size_t kc);
    ScannedDataset1D(const ScannedDataset1D& other);
    ScannedDataset1D& operator=(const ScannedDataset1D& other);
    ~ScannedDataset1D() override = default;
    size_t getSc();
    virtual data_t operator()(size_t i, size_t j);
    virtual size_t getNumInstances();
    virtual size_t getNumFeatures();
    virtual size_t getRequiredMemorySize();
};


class ScannedTargets1D : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;
    size_t s;
public:
    ScannedTargets1D(data_t* data, size_t n_instances, size_t sc);
    ScannedTargets1D(const ScannedTargets1D& other);
    ScannedTargets1D& operator=(const ScannedTargets1D& other);
    ~ScannedTargets1D() override = default;
    virtual data_t operator[](const size_t i);
    virtual size_t getNumInstances() { return n_rows; }
    virtual target_t* getValues() { return data; }
};


class MultiGrainedScanner1D : public Layer {
private:
    size_t kc; // Kernel width
public:
    MultiGrainedScanner1D(LayerConfig lconfig, size_t kc);
    ~MultiGrainedScanner1D() = default;
    virtual vdataset_p virtualize(MDDataset dataset);
    virtual vtargets_p virtualizeTargets(Labels<target_t>* targets);
    virtual size_t getRequiredMemorySize();
    virtual std::string getType() { return std::string("MultiGrainedScanner1D"); }
};

#endif // SCANNER1D_HPP_