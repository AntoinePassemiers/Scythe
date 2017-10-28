/**
    utils.hpp
    Global use parameters
    
    @author Antoine Passemiers
    @version 1.0 24/06/2017
*/

#include <cassert>
#include <cstdint>

#include "numpy/npy_common.h"
#include "numpy/noprefix.h"

#include "exceptions.hpp"

#ifndef UTILS_HPP
#define UTILS_HPP


namespace scythe {

typedef uint8_t BYTE;

constexpr int NPY_BOOL_NUM     =  0;
constexpr int NPY_INT8_NUM     =  1;
constexpr int NPY_UINT8_NUM    =  2;
constexpr int NPY_INT16_NUM    =  3;
constexpr int NPY_UINT16_NUM   =  4;
constexpr int NPY_INT32_NUM    =  7;
constexpr int NPY_UINT32_NUM   =  8;
constexpr int NPY_INT64_NUM    =  9;
constexpr int NPY_UINT64_NUM   = 10;
constexpr int NPY_FLOAT32_NUM  = 11;
constexpr int NPY_FLOAT64_NUM  = 12;
constexpr int NPY_FLOAT16_NUM  = 23;


struct Parameters {
    size_t n_jobs = 1;
    int print_layer_info = 1;
};

extern Parameters parameters;

} //namespace

#endif // UTILS_HPP
