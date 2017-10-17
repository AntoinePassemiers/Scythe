/**
    bagging.cpp
    Bootstrap aggregation
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#include "bagging.hpp"


namespace scythe {

size_t randomInstance(size_t n_instances) {
    return rand() % n_instances;
}

std::vector<size_t> randomSet(size_t n, size_t upper_bound) {
    std::vector<size_t> id_set;
    for (unsigned int i = 0; i < n; i++) {
        id_set.push_back(randomInstance(upper_bound));
    }
    return id_set;
}

std::shared_ptr<size_t> createSubsetWithReplacement(size_t n_instances, float bagging_fraction) {
    /**
        Warning : one instance cannot be used twice inside a same tree
    */
    size_t m = static_cast<size_t>(bagging_fraction * n_instances);
    if (bagging_fraction >= 1.0) m = n_instances;
    size_t* subset = static_cast<size_t*>(malloc(n_instances * sizeof(size_t)));
    // Bagging is causing segfault because n_instances is computed on the basis of
    // the total number of instances, and not the number of instances in the bag
    // TODO : pass the right number of instances to the CART algorithm
    // for (unsigned int i = 0; i < n_instances; i++) { subset[i] = OUT_OF_BAG; }
    for (unsigned int i = 0; i < n_instances; i++) { subset[i] = USED_IN_BAG; }
    for (unsigned int i = 0; i < m; i++) {
        size_t instance_id = randomInstance(n_instances);
        subset[instance_id] = USED_IN_BAG;
    }
    return std::shared_ptr<size_t>(subset);
}

}