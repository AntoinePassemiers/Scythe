/**
    utils.hpp
    Global use parameters
    
    @author Antoine Passemiers
    @version 1.0 24/06/2017
*/

#ifndef UTILS_HPP
#define UTILS_HPP


namespace scythe {

struct Parameters {
    size_t n_jobs = 1;
    int print_layer_info = 1;
};

extern Parameters parameters;

}

#endif // UTILS_HPP