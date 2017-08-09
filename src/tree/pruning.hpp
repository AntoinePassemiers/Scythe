/**
    pruning.hpp
    Post-pruning on classification and regression trees

    @author Antoine Passemiers
    @version 1.0 09/08/2017
*/

#ifndef PRUNING_HPP_
#define PRUNING_HPP_

#include <cassert>
#include <math.h>
#include <numeric>
#include <queue>
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>

#include "../tree/cart.hpp"


namespace scythe {


struct NodeLevel {
    Node* owner;
    size_t level;
};

size_t cut(Node* node);

void prune(Tree* tree, size_t max_depth);

} // namespace

#endif // PRUNING_HPP_