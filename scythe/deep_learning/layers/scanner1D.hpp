/**
    scanner1D.hpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#ifndef SCANNER1D_HPP_
#define SCANNER1D_HPP_

#include "../../misc/sets.hpp"
#include "layer.hpp"


class ScannedDataset1D : public VirtualDataset {
private:
    size_t N;  // Number of instances
    size_t M;  // Number of features
    size_t kc; // Kernel width
    size_t sc; // Number of kernel positions per column

    size_t Nprime; // Number of instances after scanning
    size_t Mprime; // Number of features after scanning
public:
    ~ScannedDataset1D() = default;
    data_t operator()(size_t i, size_t j);
};

#endif // SCANNER1D_HPP_