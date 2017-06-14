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


constexpr size_t MAX_N_DIMS = 7;

typedef unsigned int uint;
typedef double data_t;
typedef double target_t;

class VirtualDataset; // Forward declaration

typedef std::shared_ptr<VirtualDataset> vdataset_p;

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
};

// Target values
template <typename T>
struct Labels {
    T* data;
    size_t n_rows;
};


class VirtualDataset {
public:
    VirtualDataset() {};
    virtual ~VirtualDataset() = default;
    virtual data_t operator()(const size_t i, const size_t j) = 0;
    virtual size_t getNumInstances() = 0;
    virtual size_t getNumFeatures() = 0;
    virtual size_t getRequiredMemorySize() = 0;
};


template <typename T>
class AbstractTargets {
private:
    T* data;
public: 
    virtual T operator()(const size_t i) = 0;
    virtual ~AbstractTargets() = default;
};


class DirectDataset : public VirtualDataset {
private:
    data_t* data;
    size_t n_rows;
    size_t n_cols;
public:
    DirectDataset(Dataset dataset);
    DirectDataset(data_t* data, size_t n_instances, size_t n_features);
    DirectDataset(const DirectDataset& other);
    DirectDataset& operator=(const DirectDataset& other);
    ~DirectDataset() = default;
    data_t operator()(const size_t i, const size_t j);
    size_t getNumInstances() { return n_rows; }
    size_t getNumFeatures() { return n_cols; }
    size_t getRequiredMemorySize() { return n_rows * n_cols; }
};

#endif // SETS_HPP_