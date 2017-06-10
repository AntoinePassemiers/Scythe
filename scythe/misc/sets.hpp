/**
    sets.cpp
    Datasets' structures
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#ifndef SETS_HPP_
#define SETS_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

typedef unsigned int uint;
typedef double data_t;
typedef double target_t;

// Data samples without target values
struct Dataset {
    data_t* data;
    size_t n_rows;
    size_t n_cols;
};

// Target values
template <typename T>
struct Labels {
    T* data;
    size_t n_rows;
};


class AbstractDataset {
public:
    AbstractDataset() {};
    virtual ~AbstractDataset() = default;
    virtual data_t operator()(const size_t i, const size_t j) = 0;
};

template <typename T>
class AbstractTargets {
private:
    T* data;
public: 
    virtual T operator()(const size_t i) = 0;
    virtual ~AbstractTargets() = default;
};


class DirectDataset : public AbstractDataset {
private:
    data_t* data;
    size_t n_rows;
    size_t n_cols;
public:
    DirectDataset(Dataset dataset);
    ~DirectDataset() = default;
    data_t operator()(const size_t i, const size_t j);
};

#endif // SETS_HPP_