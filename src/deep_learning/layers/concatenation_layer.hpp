/**
    concatenation_layer.hpp
    Base layer with concatenation of two feature matrices

    @author Antoine Passemiers
    @version 1.0 23/06/2017
*/

#ifndef CONCATENATION_LAYER_HPP_
#define CONCATENATION_LAYER_HPP_

#include "layer.hpp"


namespace scythe {

class ConcatenationDataset : public VirtualDataset {    
private:
    proba_t* data;
    size_t n_instances;
    size_t n_virtual_cols;
    size_t stride;
    int dtype;

    // Iterator cursor
    size_t iterator_cursor;
public:
    ConcatenationDataset(size_t n_instances, size_t n_virtual_features);
    ConcatenationDataset(const ConcatenationDataset& other) = default;
    ConcatenationDataset& operator=(const ConcatenationDataset& other) = default;
    ~ConcatenationDataset() override = default;
    void concatenate(float* new_data, size_t width);
    void reset() { this->stride = 0; }
    virtual data_t operator()(const size_t i, const size_t j);
    virtual void allocateFromSampleMask(size_t* const mask, size_t, size_t, size_t, size_t);

    // Virtual iterator
    virtual void _iterator_begin(const size_t j);
    virtual void _iterator_inc();
    virtual data_t _iterator_deref();

    // Getters
    virtual size_t getNumInstances() { return n_instances; }
    virtual size_t getNumFeatures() { return stride; }
    virtual size_t getNumVirtualInstancesPerInstance() { return 1; }
    virtual int getDataType() { return dtype; }
};


class CascadeLayer : public Layer {
public:
    CascadeLayer(LayerConfig lconfig);
    ~CascadeLayer() {}
    virtual vdataset_p virtualize(MDDataset dataset);
    virtual vtargets_p virtualizeTargets(Labels* targets);
    virtual size_t getNumVirtualFeatures();
    virtual bool isConcatenable() { return true; }
    virtual std::string getType() { return std::string("CascadeLayer"); }
};

}

#endif // CONCATENATION_LAYER_HPP_