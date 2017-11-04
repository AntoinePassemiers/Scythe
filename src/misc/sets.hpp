/**
    sets.cpp
    Virtual datasets
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#ifndef SETS_HPP_
#define SETS_HPP_

#include <iostream>
#include <memory>
#include <vector>
#include <cstring>
#include <string>

#include "exceptions.hpp"
#include "utils.hpp"


namespace scythe {

constexpr size_t MAX_N_DIMS = 7;

constexpr int DTYPE_PROBA   = 0;
constexpr int DTYPE_DOUBLE  = 1;
constexpr int DTYPE_UINT_8  = 2;


typedef unsigned int uint;
typedef double       data_t;
typedef float        fast_data_t;
typedef double       target_t;
typedef float        proba_t;
typedef int          label_t;

// Forward declarations
class VirtualDataset;
class VirtualTargets;
/*
C++ note: Neither alias declarations or forward declarations work with
nested classes since the compiler can't be sure that a nested class
actually exists.
*/

typedef std::shared_ptr<VirtualDataset> vdataset_p;
typedef std::shared_ptr<VirtualTargets> vtargets_p;

// Data samples without target values
struct Dataset {
    void*  data;
    size_t n_rows;
    size_t n_cols;
    int    dtype;
};

// Data samples contained in a multi-dimensional dataset
struct MDDataset {
    void*  data;
    size_t n_dims;
    size_t dims[MAX_N_DIMS];
    int    dtype;
};

// Target values
struct Labels {
    target_t* data;
    size_t    n_rows;
};


class VirtualDataset {
protected:
    void* contiguous_data = nullptr;
    size_t n_contiguous_items = 0;
public:
    VirtualDataset() = default;
    VirtualDataset(const VirtualDataset&) = default;
    VirtualDataset& operator=(const VirtualDataset&) = default;
    virtual ~VirtualDataset() = default;
    virtual VirtualDataset* deepcopy() = 0;
    virtual VirtualDataset* createView(void* view, size_t n_rows) = 0;
    virtual data_t operator()(const size_t i, const size_t j) = 0;
    VirtualDataset* shuffleAndCreateView(std::vector<size_t>& indexes);

    // Virtual iterator
    virtual void   _iterator_begin(const size_t j) = 0;
    virtual void   _iterator_inc() = 0;
    virtual data_t _iterator_deref() = 0;

    // Getters
    virtual size_t getNumInstances() = 0; // Number of virtual instances
    virtual size_t getNumFeatures() = 0;  // Number of virtual features
    virtual size_t getNumVirtualInstancesPerInstance() = 0;
    virtual size_t getRowStride() = 0; // Size of a swappable row
    virtual size_t getNumRows() = 0;   // Number of swappable rows
    virtual int    getDataType() = 0;
    virtual void*  getData() = 0;
    size_t getItemStride();

    virtual void allocateFromSampleMask(size_t* const mask, size_t, size_t, size_t, size_t) = 0;
    void* retrieveContiguousData() { return contiguous_data; }
};


class DirectDataset : public VirtualDataset {
private:
    void* data;
    size_t  n_rows;
    size_t  n_cols;
    int dtype;

    // Iterator cursor
    size_t iterator_cursor;
public:
    DirectDataset(Dataset dataset);
    DirectDataset(void* data, size_t n_instances, size_t n_features);
    DirectDataset(const DirectDataset& other) = default;
    DirectDataset& operator=(const DirectDataset& other) = default;
    ~DirectDataset() override = default;
    virtual VirtualDataset* deepcopy();
    virtual VirtualDataset* createView(void* view, size_t n_rows);
    virtual data_t operator()(const size_t i, const size_t j);

    // Virtual iterator
    virtual void   _iterator_begin(const size_t j);
    virtual void   _iterator_inc();
    virtual data_t _iterator_deref();

    virtual void allocateFromSampleMask(size_t* const mask, size_t, size_t, size_t, size_t);

    // Getters
    virtual size_t getNumInstances() { return n_rows; }
    virtual size_t getNumFeatures() { return n_cols; }
    virtual size_t getNumVirtualInstancesPerInstance() { return 1; }
    virtual size_t getRowStride() { return n_cols; }
    virtual size_t getNumRows() { return n_rows; }
    virtual int    getDataType() { return dtype; }
    virtual void*  getData() { return data; }
};


class VirtualTargets {
protected:
    label_t* contiguous_labels = nullptr;
    size_t n_contiguous_items = 0;
public:
    VirtualTargets() {};
    VirtualTargets(const VirtualTargets&) = default;
    VirtualTargets& operator=(const VirtualTargets&) = default;
    virtual ~VirtualTargets() = default;
    virtual VirtualTargets* deepcopy() = 0;
    virtual VirtualTargets* createView(void* view, size_t n_rows) = 0;
    virtual target_t operator[](const size_t i) = 0;
    virtual size_t getNumInstances() = 0;
    virtual target_t* getValues() = 0;
    VirtualTargets* shuffleAndCreateView(std::vector<size_t>& indexes);

    // Virtual iterator
    virtual void   _iterator_begin() = 0;
    virtual void   _iterator_inc() = 0;
    virtual data_t _iterator_deref() = 0;

    virtual void allocateFromSampleMask(size_t*, size_t, size_t, size_t) = 0;
    label_t* retrieveContiguousData() { return contiguous_labels; }
};


class DirectTargets : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;

    // Iterator cursor
    size_t iterator_cursor;
public:
    DirectTargets(target_t* data, size_t n_instances);
    DirectTargets(const DirectTargets& other) = default;
    DirectTargets& operator=(const DirectTargets& other) = default;
    ~DirectTargets() override = default;
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

}  // namespace

#endif // SETS_HPP_
