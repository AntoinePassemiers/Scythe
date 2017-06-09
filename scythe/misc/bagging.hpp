/**
    bagging.hpp
    Bootstrap aggregation
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#ifndef BAGGING_HPP_
#define BAGGING_HPP_

#include <stdlib.h>
#include <memory>
#include <limits>
#include <cassert>


constexpr size_t USED_IN_BAG = 0;
constexpr size_t OUT_OF_BAG  = std::numeric_limits<int>::max();


size_t randomInstance(size_t);

std::shared_ptr<size_t> createSubsetWithReplacement(size_t, size_t);

#endif // BAGGING_HPP_