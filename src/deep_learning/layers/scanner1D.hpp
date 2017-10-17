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

    void* data; // Pointer to the raw data
    int dtype;    // Raw data type

    // Iterator cursors

public:
    ScannedDataset1D(void* data, size_t N, size_t M, size_t kc, int dtype);
    ScannedDataset1D(const ScannedDataset1D& other) = default;
    ScannedDataset1D& operator=(const ScannedDataset1D& other) = default;
    ~ScannedDataset1D() override = default;
    virtual VirtualDataset* deepcopy();
    virtual VirtualDataset* createView(void* view, size_t n_rows);
    virtual data_t operator()(size_t i, size_t j);

    // Virtual iterator
    virtual void _iterator_begin(const size_t j);
    virtual void _iterator_inc();
    virtual data_t _iterator_deref();

    // Getters
    size_t getSc() { return sc; }
    virtual size_t getNumInstances() { return Nprime; }
    virtual size_t getNumFeatures() { return Mprime; }
    virtual size_t getNumVirtualInstancesPerInstance() { return sc; }
    virtual size_t getNumRows() { return N; }
    virtual size_t getRowStride() { return M; }
    virtual int    getDataType() { return dtype; }
    virtual void*  getData() { return data; }

    virtual void allocateFromSampleMask(size_t* const mask, size_t, size_t, size_t, size_t);
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
    virtual VirtualTargets* deepcopy();
    virtual VirtualTargets* createView(void* view, size_t n_rows);
    virtual target_t operator[](const size_t i);
    virtual size_t getNumInstances() { return n_rows; }
    virtual target_t* getValues() { return data; }

    // Virtual iterator
    virtual void   _iterator_begin();
    virtual void   _iterator_inc();
    virtual data_t _iterator_deref();

    virtual void allocateFromSampleMask(size_t*, size_t, size_t, size_t);
};


class MultiGrainedScanner1D : public Layer {
private:
    size_t kc; // Kernel width
public:
    MultiGrainedScanner1D(LayerConfig lconfig, size_t kc);
    ~MultiGrainedScanner1D() {}
    virtual vdataset_p virtualize(MDDataset dataset);
    virtual vtargets_p virtualizeTargets(Labels* targets);
    virtual size_t getNumVirtualFeatures();
    virtual bool isConcatenable() { return false; }
    virtual std::string getType() { return std::string("MultiGrainedScanner1D"); }
};

} // namespace

#endif // SCANNER1D_HPP_