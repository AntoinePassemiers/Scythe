/**
    grayscale.hpp
    Compute pixel value densities

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#ifndef GRAYSCALE_HPP
#define GRAYSCALE_HPP

#include "density.hpp"


namespace scythe {

Density* getArbitraryPixelDensities(size_t n_features, size_t n_classes);

}

#endif // GRAYSCALE_HPP