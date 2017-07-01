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
    proba_t* data;
    size_t n_instances;
    size_t n_virtual_cols;
    size_t stride;
    int dtype;
public:
    template<typename T>
    class Iterator : public VirtualDataset::Iterator<T> {
    private:
        size_t cursor;
        size_t n_virtual_cols;
        T* data;
    public:
        Iterator(T* data, size_t n_virtual_cols) : 
            cursor(0), n_virtual_cols(n_virtual_cols), data(data) {}
        ~Iterator() = default;
        T operator*() { return data[cursor]; }
        Iterator& operator++();
        Iterator& operator--();
    };
    ConcatenationDataset(size_t n_instances, size_t n_virtual_features);
    ConcatenationDataset(const ConcatenationDataset& other) = default;
    ConcatenationDataset& operator=(const ConcatenationDataset& other) = default;
    ~ConcatenationDataset() override = default;
    void concatenate(float* new_data, size_t width);
    virtual data_t operator()(const size_t i, const size_t j);
    virtual std::shared_ptr<void> _operator_ev(const size_t j); // Type erasure
    virtual size_t getNumInstances() { return n_instances; }
    virtual size_t getNumFeatures() { return stride; }
    virtual size_t getRequiredMemorySize() { return n_instances * n_virtual_cols; }
    virtual size_t getNumVirtualInstancesPerInstance() { return 1; }
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


template<typename T>
ConcatenationDataset::Iterator<T>& ConcatenationDataset::Iterator<T>::operator++() {
    cursor += n_virtual_cols;
    return *this;
}

template<typename T>
ConcatenationDataset::Iterator<T>& ConcatenationDataset::Iterator<T>::operator--() {
    cursor -= n_virtual_cols;
    return *this;
}

#endif // CONCATENATION_LAYER_HPP_