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

namespace scythe {

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
    int dtype;    // Raw data type
public:
    template<typename T>
    class Iterator : public VirtualDataset::Iterator<T> {
    private:
        size_t x;
        size_t i;
        size_t q;
        size_t M;
        size_t P;
        size_t sc;
        size_t sr;
        T* data;
    public:
        Iterator(T* data, size_t x, size_t M, size_t P, size_t sc, size_t sr) : 
            x(x), i(0), q(0), M(M), P(P), sc(sc), sr(sr) {}
        Iterator(const Iterator&) = default;
        Iterator& operator=(const Iterator&) = default;
        ~Iterator() = default;
        T operator*();
        Iterator& operator++();
    };
    ScannedDataset2D(data_t* data, size_t N, size_t M, 
        size_t P, size_t kc, size_t kr, int dtype);
    ScannedDataset2D(const ScannedDataset2D& other) = default;
    ScannedDataset2D& operator=(const ScannedDataset2D& other) = default;
    ~ScannedDataset2D() override = default;
    size_t getSc() { return sc; }
    size_t getSr() { return sr; }
    virtual data_t operator()(size_t i, size_t j);
    virtual std::shared_ptr<void> _operator_ev(const size_t j); // Type erasure
    virtual size_t getNumInstances() { return Nprime; }
    virtual size_t getNumFeatures() { return Mprime; }
    virtual size_t getRequiredMemorySize() { return Nprime * Mprime; }
    virtual size_t getNumVirtualInstancesPerInstance() { return sc * sr; }
    virtual int getDataType() { return dtype; }
};


class ScannedTargets2D : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;
    size_t s;
public:
    ScannedTargets2D(target_t* data, size_t n_instances, size_t sc, size_t sr);
    ScannedTargets2D(const ScannedTargets2D& other) = default;
    ScannedTargets2D& operator=(const ScannedTargets2D& other) = default;
    ~ScannedTargets2D() override = default;
    virtual target_t operator[](const size_t i);
    virtual size_t getNumInstances() { return n_rows; }
    virtual target_t* getValues() { return data; }
};


class MultiGrainedScanner2D : public Layer {
private:
    size_t kc; // Kernel width
    size_t kr; // Kernel height
public:
    MultiGrainedScanner2D(LayerConfig lconfig, size_t kc, size_t kr);
    ~MultiGrainedScanner2D() {}
    virtual vdataset_p virtualize(MDDataset dataset);
    virtual vtargets_p virtualizeTargets(Labels* targets);
    virtual size_t getRequiredMemorySize();
    virtual size_t getNumVirtualFeatures();
    virtual bool isConcatenable() { return false; }
    virtual std::string getType() { return std::string("MultiGrainedScanner2D"); }
};


template<typename T>
T ScannedDataset2D::Iterator<T>::operator*() {
    return data[x + i * P + q / sr];
}

template<typename T>
ScannedDataset2D::Iterator<T>& ScannedDataset2D::Iterator<T>::operator++() {
    ++i;
    if (i == sc) {
        ++q;
        i = 0;
        if (q == sr) {
            q = 0;
            x += (M * P);
        }
    }
    return *this;
}

}

#endif // SCANNER2D_HPP_