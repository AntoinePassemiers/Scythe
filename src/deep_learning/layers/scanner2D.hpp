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

    void* data; // Pointer to the raw data
    int dtype;    // Raw data type

    // Iterator cursors
    size_t _it_x;
    size_t _it_i;
    size_t _it_q;
public:
    ScannedDataset2D(void* data, size_t N, size_t M, 
        size_t P, size_t kc, size_t kr, int dtype);
    ScannedDataset2D(const ScannedDataset2D& other) = default;
    ScannedDataset2D& operator=(const ScannedDataset2D& other) = default;
    ~ScannedDataset2D() override = default;
    virtual VirtualDataset* deepcopy();
    virtual data_t operator()(size_t i, size_t j);

    // Virtual iterator
    virtual void _iterator_begin(const size_t j);
    virtual void _iterator_inc();
    virtual data_t _iterator_deref();

    template<typename T, typename fast_T>
    void generic_allocateFromSampleMask(size_t* const mask, size_t, size_t, size_t, size_t);
    virtual void allocateFromSampleMask(size_t* const mask, size_t, size_t, size_t, size_t);

    // Getters
    size_t getSc() { return sc; }
    size_t getSr() { return sr; }
    virtual size_t getNumInstances() { return Nprime; }
    virtual size_t getNumFeatures() { return Mprime; }
    virtual size_t getNumVirtualInstancesPerInstance() { return sc * sr; }
    virtual size_t getNumRows() { return N; }
    virtual size_t getRowStride() { return M * P; }
    virtual int    getDataType() { return dtype; }
    virtual void*  getData() { return data; }
};


class ScannedTargets2D : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;
    size_t sc;
    size_t sr;
    size_t s;
    size_t _it_x;
    size_t _it_i;
public:
    ScannedTargets2D(target_t* data, size_t n_instances, size_t sc, size_t sr);
    ScannedTargets2D(const ScannedTargets2D& other) = default;
    ScannedTargets2D& operator=(const ScannedTargets2D& other) = default;
    ~ScannedTargets2D() override = default;
    virtual VirtualTargets* deepcopy();
    virtual target_t operator[](const size_t i);
    virtual size_t getNumInstances() { return n_rows; }
    virtual target_t* getValues() { return data; }

    // Virtual iterator
    virtual void _iterator_begin();
    virtual void _iterator_inc();
    virtual data_t _iterator_deref();

    virtual void allocateFromSampleMask(size_t*, size_t, size_t, size_t);
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
    virtual size_t getNumVirtualFeatures();
    virtual bool isConcatenable() { return false; }
    virtual std::string getType() { return std::string("MultiGrainedScanner2D"); }
};


template<typename T, typename fast_T>
void ScannedDataset2D::generic_allocateFromSampleMask(
    size_t* const sample_mask, size_t node_id, size_t feature_id, 
    size_t n_items, size_t n_instances) {

    T* t_data = static_cast<T*>(data);
    fast_T* t_contiguous_data = static_cast<fast_T*>(contiguous_data);
    if (n_items != this->n_contiguous_items) { // TODO
        if (contiguous_data != nullptr) {
            delete[] contiguous_data;
        }
        t_contiguous_data = new fast_T[n_items];
        this->n_contiguous_items = n_items;
    }

    uint k = 0;
    _it_x = P * (feature_id % kc) + (feature_id / kr);
    _it_i = 0;
    _it_q = 0;
    for (uint i = 0; i < n_instances; i++) {
        if (sample_mask[i] == node_id) {
            t_contiguous_data[k++] = static_cast<fast_T>(t_data[_it_x + _it_i + _it_q]);
        }
        _it_i++;
        if (_it_i == sc) {
            _it_q += M;
            _it_i = 0;
            if (_it_q == sr * M) {
                _it_q = 0;
                _it_x += (M * P);
            }
        }
    }
    contiguous_data = static_cast<void*>(t_contiguous_data);
    assert(k == n_items);
}



} // namespace

#endif // SCANNER2D_HPP_