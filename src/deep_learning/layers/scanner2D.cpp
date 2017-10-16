/**
    scanner2D.cpp
    Multi-grained scanning

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "scanner2D.hpp"

namespace scythe {

Parameters parameters;

ScannedDataset2D::ScannedDataset2D(
    void* data, size_t N, size_t M, size_t P, size_t kc, size_t kr, int dtype) : 
    N(N),                // Number of instances
    M(M),                // Instance height
    P(P),                // Instance width
    kc(kc),              // Kernel width
    kr(kr),              // Kernel height
    sc(P - kc + 1),      // Number of kernel positions per column
    sr(M - kr + 1),      // Number of kernel positions per row
    Nprime(N * sr * sc), // Number of instances after scanning
    Mprime(kc * kr),     // Number of features after scanning
    data(data),          // Pointer to the raw data
    dtype(dtype) {       // Raw data type
    if (parameters.print_layer_info) {
        #ifdef _OMP
            #pragma omp critical(scanned_dataset_2d_display_info)
        #endif
        {
            std::cout << "\tKernel width  : " << kc << std::endl;
            std::cout << "\tKernel height : " << kr << std::endl;
            std::cout << "\tN prime       : " << Nprime << std::endl;
            std::cout << "\tM prime       : " << Mprime << std::endl;
        }
    }
}

VirtualDataset* ScannedDataset2D::deepcopy() {
    size_t n_required_bytes = getNumRows() * getRowStride() * getItemStride();
    void* new_data = malloc(n_required_bytes);
    std::memcpy(new_data, data, n_required_bytes);
    return new ScannedDataset2D(new_data, N, M, P, kc, kr, dtype);
}

void ScannedDataset2D::allocateFromSampleMask(
    size_t* const sample_mask, size_t node_id, size_t feature_id, size_t n_items, size_t n_instances) {
    /**
        Allocate memory for storing temporary values of a single feature,
        for the data samples belonging to the current node.
        This method is called right before the inner loop of the CART algorithm,
        and its purpose is to avoid calling virtual functions inside the vectorized
        inner loop.

        @param sample_mask
            Pointer indicating for each data sample the id of the node it belongs to
        @param node_id
            Id of the current node
        @param feature_id
            Id of the attribute whose values are going to be stored
        @param n_items
            Number of data samples belonging to the current node
        @param n_instances
            Number of data samples in the whole dataset
    */
    /**
    fast_data_t* t_contiguous_data = static_cast<fast_data_t*>(contiguous_data);
    if (n_items != this->n_contiguous_items) { // TODO
        if (contiguous_data != nullptr) {
            delete[] contiguous_data;
        }
        t_contiguous_data = new fast_data_t[n_items];
        this->n_contiguous_items = n_items;
    }

    uint k = 0;
    _it_x = P * (feature_id % kc) + (feature_id / kr);
    _it_i = 0;
    _it_q = 0;
    for (uint i = 0; i < n_instances; i++) {
        if (sample_mask[i] == node_id) {
            t_contiguous_data[k++] = static_cast<fast_data_t>(static_cast<data_t*>(data)[_it_x + _it_i + _it_q]);
        }
        _it_i++;
        if (_it_i == sc) {
            _it_q += M;
            _it_i = 0;
            if (_it_q == sr * M) {
                _it_q = 0;
                _it_x += (M * P);
            }
        }
    }
    contiguous_data = static_cast<void*>(t_contiguous_data);
    assert(k == n_items);
    */

    switch (getDataType()) {
        case NPY_UINT8_NUM:
            generic_allocateFromSampleMask<npy_uint8, npy_uint8>(
                sample_mask, node_id, feature_id, n_items, 
                n_instances);
            break;
        default:
            generic_allocateFromSampleMask<data_t, fast_data_t>(
                sample_mask, node_id, feature_id, n_items, 
                n_instances);
            break;
    }


}

data_t ScannedDataset2D::operator()(size_t i, size_t j) {
    size_t n = i / (sr * sc);
    size_t m = (i % sc) + (j % kc);
    size_t p = ((i % (sr * sc)) / sr) + (j / kr);
    switch (getDataType()) {
        case NPY_UINT8_NUM:
            return static_cast<uint8_t*>(data)[(M * P) * n + (P * m) + p];
        default:
            return static_cast<data_t*>(data)[(M * P) * n + (P * m) + p];
    }
}

void ScannedDataset2D::_iterator_begin(const size_t j) {
    _it_x = P * (j % kc) + (j / kr);
    _it_i = 0;
    _it_q = 0;
}

void ScannedDataset2D::_iterator_inc() {
    _it_i++;
    if (_it_i == sc) {
        _it_q += M;
        _it_i = 0;
        if (_it_q == sr * M) {
            _it_q = 0;
            _it_x += (M * P);
        }
    }
}

data_t ScannedDataset2D::_iterator_deref() {
    switch (getDataType()) {
        case NPY_UINT8_NUM:
            return static_cast<uint8_t*>(data)[_it_x + _it_i + _it_q];
        default:
            return static_cast<data_t*>(data)[_it_x + _it_i + _it_q];
    }
}

ScannedTargets2D::ScannedTargets2D(target_t* data, size_t n_instances, size_t sc, size_t sr) :
    VirtualTargets::VirtualTargets(), data(data), n_rows(n_instances), sc(sc), sr(sr), s(sc * sr) {}

VirtualTargets* ScannedTargets2D::deepcopy() {
    size_t n_required_bytes = getNumInstances() * sizeof(target_t);
    target_t* new_data = static_cast<target_t*>(malloc(n_required_bytes));
    std::memcpy(new_data, data, n_required_bytes);
   return new ScannedTargets2D(new_data, n_rows, sc, sr);
}

void ScannedTargets2D::allocateFromSampleMask(
    size_t* sample_mask, size_t node_id, size_t n_items, size_t n_instances) {
    /**
        Allocate memory for storing temporary values of the labels,
        for the data samples belonging to the current node.
        This method is called right before the inner loop of the CART algorithm,
        and its purpose is to avoid calling virtual functions inside the vectorized
        inner loop.

        @param sample_mask
            Pointer indicating for each data sample the id of the node it belongs to
        @param node_id
            Id of the current node
        @param n_items
            Number of data samples belonging to the current node
        @param n_instances
            Number of data samples in the whole dataset
    */
    if (n_items != this->n_contiguous_items) { // TODO
        if (contiguous_labels != nullptr) {
            delete[] contiguous_labels;
        }
        contiguous_labels = new label_t[n_items];
        this->n_contiguous_items = n_items;
    }
    uint k = 0;
    _it_x = _it_i = 0;
    for (uint i = 0; i < n_instances; i++) {
        if (sample_mask[i] == node_id) {
            contiguous_labels[k++] = data[_it_i];
        }
        if (++_it_x == s) {
            _it_i++;
            _it_x = 0;
        }
    }
    assert(k == n_items);
}

void ScannedTargets2D::_iterator_begin() {
    _it_x = _it_i = 0;
}

void ScannedTargets2D::_iterator_inc() {
    if (++_it_x == s) {
        _it_i++;
        _it_x = 0;
    }
}

target_t ScannedTargets2D::_iterator_deref() {
    return data[_it_i];
}

target_t ScannedTargets2D::operator[](const size_t i) {
    return data[i / s];
}

MultiGrainedScanner2D::MultiGrainedScanner2D(
    LayerConfig lconfig, size_t kc, size_t kr) : Layer(lconfig), kc(kc), kr(kr) {}

vdataset_p MultiGrainedScanner2D::virtualize(MDDataset dataset) {
    assert(dataset.n_dims == 3);
    assert(dataset.dims[0] > 0);
    assert(dataset.dims[1] > 0);
    assert(dataset.dims[2] > 0);
    Layer::vdataset = std::shared_ptr<ScannedDataset2D>(
        new ScannedDataset2D(
            static_cast<data_t*>(dataset.data), // TODO : type erasure
            dataset.dims[0],
            dataset.dims[1],
            dataset.dims[2],
            this->kc,
            this->kr,
            dataset.dtype));
    return Layer::vdataset;
}

vtargets_p MultiGrainedScanner2D::virtualizeTargets(Labels* targets) {
    ScannedDataset2D* vdataset = dynamic_cast<ScannedDataset2D*>((this->vdataset).get());
    size_t sc = vdataset->getSc();
    size_t sr = vdataset->getSr();
    size_t n_rows = vdataset->getNumRows();
    assert(sc > 0);
    assert(sr > 0);
    return std::shared_ptr<ScannedTargets2D>(
        new ScannedTargets2D(targets->data, n_rows, sc, sr));
}

size_t MultiGrainedScanner2D::getNumVirtualFeatures() {
    ScannedDataset2D* sdataset = dynamic_cast<ScannedDataset2D*>(vdataset.get());
    size_t n_vfeatures = sdataset->getSc() * sdataset->getSr();
    assert(n_vfeatures > 0);
    if (lconfig.fconfig.task == CLASSIFICATION_TASK) {
        n_vfeatures *= lconfig.fconfig.n_classes;
    }
    return n_vfeatures * lconfig.n_forests;
}

} // namespace