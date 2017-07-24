/**
    density.hpp
    Basic density structure of a set of features

    @author Antoine Passemiers
    @version 1.3 12/04/2017
*/

#ifndef DENSITY_HPP_
#define DENSITY_HPP_

#include <cassert>
#include <math.h>
#include <numeric>
#include <queue>
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>
// #include <algorithm.h>

#include "../misc/sets.hpp"


namespace scythe {

// Partitioning of the input's density function
constexpr int QUARTILE_PARTITIONING   = 0xB23A40;
constexpr int DECILE_PARTITIONING     = 0xB23A41;
constexpr int PERCENTILE_PARTITIONING = 0xB23A42;

struct Density {
    bool    is_categorical;
    data_t  split_value;
    data_t* values;
    size_t  n_values;
    size_t* counters_left;
    size_t* counters_right;
    size_t* counters_nan;
};

}

#endif // DENSITY_HPP