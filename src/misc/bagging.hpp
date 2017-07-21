/**
    bagging.hpp
    Bootstrap aggregation
    
    @author Antoine Passemiers
    @version 1.0 09/06/2017
*/

#ifndef BAGGING_HPP_
#define BAGGING_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <memory>
#include <limits>
#include <cassert>


namespace scythe {

constexpr size_t USED_IN_BAG = 0;
constexpr size_t OUT_OF_BAG  = std::numeric_limits<size_t>::max();


size_t randomInstance(size_t);

std::shared_ptr<size_t> createSubsetWithReplacement(size_t, size_t);

}

#endif // BAGGING_HPP_