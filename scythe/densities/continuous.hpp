/**
    continuous.hpp
    Compute densities of continuous variables

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#ifndef CONTINUOUS_HPP
#define CONTINUOUS_HPP

#include "density.hpp"


namespace scythe {

Density* computeDensities(VirtualDataset* data, size_t n_classes, data_t nan_value, int partitioning);

}

#endif // CONTINUOUS_HPP