/**
    proba.hpp
    Compute standard probability densities

    @author Antoine Passemiers
    @version 1.0 25/06/2017
*/

#ifndef PROBA_HPP
#define PROBA_HPP

#include "density.hpp"


namespace scythe {

Density* getArbitraryProbaDensities(size_t n_features, size_t n_classes);

}

#endif // PROBA_HPP