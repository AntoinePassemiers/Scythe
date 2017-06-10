/**
    layer.hpp
    Deep learning base layer

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#ifndef FOREST_HPP_
#define FOREST_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <memory>
#include <limits>
#include <cassert>

constexpr size_t DATASET_N_DIMENSIONS = 2;
constexpr size_t MAX_N_DIMENSIONS     = 7;

/**
    Main goal of layers: ensuring that each forest gets
    a two-dimensional dataset as input, and ensuring that
    the dimensionality of the output is right
    (1d for regression, 2d for classification). These dimensionalities
    must be invariant to the complexity of cascades and convolutional layers.

    Therefore, the shapes of the datasets are "re-mapped" between layers, and
    the definition of how it works must be defined in each layer class.
*/

class Layer {
private:
    // The product of bmap_in_shape elements must be equal to the product of
    // amap_in_shape elements. Similarly, the product of bmap_out_shape elements
    // must be equal to the product of amap_out_shape elements.
    size_t bmap_in_shape[DATASET_N_DIMENSIONS];  // Input shape before re-mapping
    size_t bmap_out_shape[DATASET_N_DIMENSIONS]; // Output shape before re-mapping
    size_t amap_in_shape[MAX_N_DIMENSIONS];  // Input shape after re-mapping
    size_t amap_out_shape[MAX_N_DIMENSIONS]; // Output shape after re-mapping
}

#endif // FOREST_HPP_