/**
    forest.cpp
    Forest abstract class and configurations

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#include "forest.hpp"


namespace scythe {

void Forest::preprocessDensities(VirtualDataset* dataset) {
    if (dataset->getDataType() == DTYPE_UINT_8) {
        /**
        this->densities = std::move(std::shared_ptr<Density>(
            getArbitraryPixelDensities(
                dataset->getNumFeatures(), base_tree_config.n_classes)));
        */
        this->densities = std::move(std::shared_ptr<Density>(computeDensities(
            dataset, base_tree_config.n_classes, base_tree_config.nan_value, base_tree_config.partitioning)));
    }
    else if (dataset->getDataType() == DTYPE_PROBA) {
        /**
        this->densities = std::move(std::shared_ptr<Density>(
            getArbitraryProbaDensities(
                dataset->getNumFeatures(), base_tree_config.n_classes)));
        */
        this->densities = std::move(std::shared_ptr<Density>(computeDensities(
            dataset, base_tree_config.n_classes, base_tree_config.nan_value, base_tree_config.partitioning)));
    }
    else {
        this->densities = std::move(std::shared_ptr<Density>(computeDensities(
            dataset, base_tree_config.n_classes, base_tree_config.nan_value, base_tree_config.partitioning)));
    }
}

void Forest::save(std::ofstream& file) {
    file << trees.size() << " ";
    file << n_instances << " ";
    file << n_features << std::endl;
    for (std::shared_ptr<Tree> tree : trees) {
        saveTree(tree.get(), file);
    }
}

void Forest::load(std::ifstream& file) {
    size_t n_trees;
    file >> n_trees >> n_instances >> n_features;
    for (size_t i = 0; i < n_trees; i++) {
        trees.push_back(std::shared_ptr<Tree>(loadTree(file, &base_tree_config)));
    }
}

} // namespace