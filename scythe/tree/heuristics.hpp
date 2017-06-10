/**
    heuristics.hpp
    Meta-heuristics for increasing CART's speed

    @author Antoine Passemiers
    @version 1.3 10/06/2017
*/

#ifndef HEURISTICS_HPP_
#define HEURISTICS_HPP_

#include <cassert>
#include <cmath>
#include <math.h>
#include <numeric>
#include <queue>
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>



void selectFeaturesToConsider(size_t* to_use, size_t n_features, size_t max_n_features);

#endif // HEURISTICS_HPP_