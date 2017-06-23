/**
    concatenation_layer.hpp
    Base layer with concatenation of two feature matrices

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#ifndef CONCATENATION_LAYER_HPP_
#define CONCATENATION_LAYER_HPP_

#include "layer.hpp"


class ConcatenationDataset : public VirtualDataset {    
private:
    data_t* data;
    size_t n_instances;
    size_t n_virtual_cols;
    size_t stride;
    int dtype;
public:
    ConcatenationDataset(size_t n_instances, size_t n_virtual_features);
    ConcatenationDataset(const ConcatenationDataset& other);
    ConcatenationDataset& operator=(const ConcatenationDataset& other);
    ~ConcatenationDataset() override = default;
    void concatenate(float* new_data, size_t width);
    virtual data_t operator()(const size_t i, const size_t j);
    virtual size_t getNumInstances() { return n_instances; }
    virtual size_t getNumFeatures() { return stride; }
    virtual size_t getRequiredMemorySize() { return n_instances * n_virtual_cols; }
    virtual int getDataType() { return dtype; }
};


class CascadeLayer : public Layer {
public:
    CascadeLayer(LayerConfig lconfig);
    ~CascadeLayer() {}
    virtual vdataset_p virtualize(MDDataset dataset);
    virtual vtargets_p virtualizeTargets(Labels<target_t>* targets);
    virtual size_t getRequiredMemorySize();
    virtual size_t getNumVirtualFeatures();
    virtual bool isConcatenable() { return true; }
    virtual std::string getType() { return std::string("CascadeLayer"); }
};

#endif // CONCATENATION_LAYER_HPP_