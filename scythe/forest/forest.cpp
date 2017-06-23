/**
    forest.cpp
    Forest abstract class and configurations

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#include "forest.hpp"


void Forest::preprocessDensities(VirtualDataset* dataset) {
    if (dataset->getDataType() == gbdf::DTYPE_UINT_8) {
        this->densities = std::move(std::shared_ptr<Density>(
            getArbitraryPixelDensities(
                dataset->getNumFeatures(), base_tree_config.n_classes)));
    }
    else {
        this->densities = std::move(std::shared_ptr<Density>(computeDensities(
            dataset, base_tree_config.n_classes, base_tree_config.nan_value, base_tree_config.partitioning)));
    }
}