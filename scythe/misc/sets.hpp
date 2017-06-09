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

struct Dataset {
    data_t* data;
    size_t n_rows;
    size_t n_cols;
};

template <typename T>
struct Labels {
    T* data;
    size_t n_rows;
};

struct GroundTruth {
    data_t* data;
    size_t n_rows;
};


#endif // SETS_HPP_