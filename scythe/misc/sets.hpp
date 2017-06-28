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


constexpr size_t MAX_N_DIMS = 7;

namespace gbdf {
    constexpr int DTYPE_PROBA   = 0;
    constexpr int DTYPE_DOUBLE  = 1;
    constexpr int DTYPE_UINT_8  = 2;
}

typedef unsigned int uint;
typedef double data_t;
typedef double target_t;
typedef float proba_t;

// Forward declarations
class VirtualDataset;
class VirtualTargets;
/*
Note: Neither alias declarations or forward declarations work with
nested classes since the compiler can't be sure that a nested class
actually exists.
*/

typedef std::shared_ptr<VirtualDataset> vdataset_p;
typedef std::shared_ptr<VirtualTargets> vtargets_p;

// Data samples without target values
struct Dataset {
    data_t* data;
    size_t n_rows;
    size_t n_cols;
};

// Data samples contained in a multi-dimensional dataset
struct MDDataset {
    data_t* data;
    size_t n_dims;
    size_t dims[MAX_N_DIMS];
    int dtype;
};

// Target values
template <typename T>
struct Labels {
    T* data;
    size_t n_rows;
};


class VirtualDataset {
public:
    template<typename T>
    class VirtualIterator {
    public:
        virtual T operator*();
        virtual VirtualIterator& operator++();
        VirtualIterator operator++(int);
        virtual VirtualIterator& operator--();
        VirtualIterator operator--(int);
    };
    VirtualDataset() {};
    virtual ~VirtualDataset() = default;
    virtual data_t operator()(const size_t i, const size_t j) = 0;
    template<typename T>
    VirtualIterator<T> operator()(const size_t j); // TODO: set as pure virtual
    virtual size_t getNumInstances() = 0;
    virtual size_t getNumFeatures() = 0;
    virtual size_t getRequiredMemorySize() = 0;
    virtual size_t getNumVirtualInstancesPerInstance() = 0;
    virtual int getDataType() = 0;
};


class DirectDataset : public VirtualDataset {
private:
    data_t* data;
    size_t n_rows;
    size_t n_cols;
    int dtype;
public:
    template<typename T>
    class VirtualIterator : public VirtualDataset::VirtualIterator<T> {
    private:
        size_t cursor;
        size_t n_cols;
        T* data;
    public:
        VirtualIterator(T* data, size_t n_cols, size_t j) : 
            VirtualDataset::VirtualIterator<T>::cursor(j), n_cols(n_cols), data(data) {}
        ~VirtualIterator() = default;
        T operator*() { return data[cursor]; }
        VirtualIterator& operator++();
        VirtualIterator& operator--();
    };
    DirectDataset(Dataset dataset);
    DirectDataset(data_t* data, size_t n_instances, size_t n_features);
    DirectDataset(const DirectDataset& other);
    DirectDataset& operator=(const DirectDataset& other);
    ~DirectDataset() override = default;
    virtual data_t operator()(const size_t i, const size_t j);
    virtual size_t getNumInstances() { return n_rows; }
    virtual size_t getNumFeatures() { return n_cols; }
    virtual size_t getRequiredMemorySize() { return n_rows * n_cols; }
    virtual size_t getNumVirtualInstancesPerInstance() { return 1; }
    virtual int getDataType() { return dtype; }
};


class VirtualTargets {
public:
    VirtualTargets() {};
    virtual ~VirtualTargets() = default;
    virtual target_t operator[](const size_t i) = 0;
    virtual size_t getNumInstances() = 0;
    virtual target_t* getValues() = 0;
};


class DirectTargets : public VirtualTargets {
private:
    target_t* data;
    size_t n_rows;
public:
    DirectTargets(data_t* data, size_t n_instances);
    DirectTargets(const DirectTargets& other);
    DirectTargets& operator=(const DirectTargets& other);
    ~DirectTargets() override = default;
    virtual data_t operator[](const size_t i);
    virtual size_t getNumInstances() { return n_rows; }
    virtual target_t* getValues() { return data; }
};

template<typename T>
VirtualDataset::VirtualIterator<T> VirtualDataset::VirtualIterator<T>::operator++(int) {
    ++(*this);
    return *this;
}

template<typename T>
VirtualDataset::VirtualIterator<T> VirtualDataset::VirtualIterator<T>::operator--(int) {
    --(*this);
    return *this;
}

template<typename T>
DirectDataset::VirtualIterator<T>& DirectDataset::VirtualIterator<T>::operator++() {
    cursor += n_cols;
    return *this;
}

template<typename T>
DirectDataset::VirtualIterator<T>& DirectDataset::VirtualIterator<T>::operator--() {
    cursor -= n_cols;
    return *this;
}

#endif // SETS_HPP_