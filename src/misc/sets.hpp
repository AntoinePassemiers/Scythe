/**
    sets.cpp
    Virtual datasets
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#ifndef SETS_HPP_
#define SETS_HPP_

#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <memory>
#include <limits>
#include <cassert>
#include <string.h>

#include "exceptions.hpp"


namespace scythe {

constexpr size_t MAX_N_DIMS = 7;

constexpr int DTYPE_PROBA   = 0;
constexpr int DTYPE_DOUBLE  = 1;
constexpr int DTYPE_UINT_8  = 2;


typedef unsigned int uint;
typedef double       data_t;
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
    void* data;
    size_t n_rows;
    size_t n_cols;
};

// Data samples contained in a multi-dimensional dataset
struct MDDataset {
    void* data;
    size_t n_dims;
    size_t dims[MAX_N_DIMS];
    int dtype;
};

// Target values
struct Labels {
    target_t* data;
    size_t n_rows;
};


class VirtualDataset {
private:
    data_t* contiguous_data = nullptr;
    size_t n_contiguous_items = 0;
public:
    VirtualDataset() = default;
    VirtualDataset(const VirtualDataset&) = default;
    VirtualDataset& operator=(const VirtualDataset&) = default;
    virtual ~VirtualDataset() = default;
    virtual data_t operator()(const size_t i, const size_t j) = 0;

    // Virtual iterator
    virtual void _iterator_begin(const size_t j) = 0;
    virtual void _iterator_inc() = 0;
    virtual data_t _iterator_deref() = 0;

    // Getters
    virtual size_t getNumInstances() = 0;
    virtual size_t getNumFeatures() = 0;
    virtual size_t getNumVirtualInstancesPerInstance() = 0;
    virtual int getDataType() = 0;

    void allocateFromSampleMask(size_t*, size_t, size_t, size_t, size_t);
    data_t* retrieveContiguousData() { return contiguous_data; }
};


class DirectDataset : public VirtualDataset {
private:
    data_t* data;
    size_t n_rows;
    size_t n_cols;
    int dtype;

    // Iterator cursor
    size_t iterator_cursor;
public:
    DirectDataset(Dataset dataset);
    DirectDataset(data_t* data, size_t n_instances, size_t n_features);
    DirectDataset(const DirectDataset& other) = default;
    DirectDataset& operator=(const DirectDataset& other) = default;
    ~DirectDataset() override = default;
    virtual data_t operator()(const size_t i, const size_t j);

    // Virtual iterator
    virtual void _iterator_begin(const size_t j);
    virtual void _iterator_inc();
    virtual data_t _iterator_deref();

    // Getters
    virtual size_t getNumInstances() { return n_rows; }
    virtual size_t getNumFeatures() { return n_cols; }
    virtual size_t getNumVirtualInstancesPerInstance() { return 1; }
    virtual int getDataType() { return dtype; }
};


class VirtualTargets {
private:
    label_t* contiguous_labels = nullptr;
    size_t n_contiguous_items = 0;
public:
    VirtualTargets() {};
    VirtualTargets(const VirtualTargets&) = default;
    VirtualTargets& operator=(const VirtualTargets&) = default;
    virtual ~VirtualTargets() = default;
    virtual target_t operator[](const size_t i) = 0;
    virtual size_t getNumInstances() = 0;
    virtual target_t* getValues() = 0;

    void allocateFromSampleMask(size_t*, size_t, size_t, size_t);
    label_t* retrieveContiguousData() { return contiguous_labels; }
};


class DirectTargets : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;
public:
    DirectTargets(target_t* data, size_t n_instances);
    DirectTargets(const DirectTargets& other) = default;
    DirectTargets& operator=(const DirectTargets& other) = default;
    ~DirectTargets() override = default;
    virtual target_t operator[](const size_t i);
    virtual size_t getNumInstances() { return n_rows; }
    virtual target_t* getValues() { return data; }
};

}

#endif // SETS_HPP_