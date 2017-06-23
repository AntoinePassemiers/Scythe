/**
    layer.hpp
    Deep learning base layer

    @author Antoine Passemiers
    @version 1.0 10/06/2017
*/

#include "layer.hpp"


LayerConfig::LayerConfig() :
     fconfig(), n_forests(0), forest_type(gbdf::COMPLETE_RANDOM_FOREST) {}

Layer::Layer(LayerConfig lconfig) :
    name(), in_shape(), virtual_in_shape(), virtual_out_shape(), 
    children(), forests(), vdataset(nullptr), vtargets(nullptr), lconfig() {
    this->lconfig = lconfig;
}

Layer::~Layer() {}

void Layer::grow(vdataset_p vdataset, vtargets_p vtargets) {
    assert(!grown);
    assert(forests.size() == 0);
    Forest* forest;
    std::cout << "AA" << std::endl;
    for (unsigned int i = 0; i < lconfig.n_forests; i++) {
        std::cout << "BB" << std::endl;
        if (lconfig.forest_type == gbdf::RANDOM_FOREST) {
            forest = new ClassificationRF(
                &lconfig.fconfig, 
                vdataset->getNumInstances(), 
                vdataset->getNumFeatures());
        }
        else if (lconfig.forest_type == gbdf::GB_FOREST) {
            // forest = new ClassificationGB(config, dataset->n_rows, dataset->n_cols);
            std::cout << "Error: gradient boosting is not supported" << std::endl;
        }
        else if (lconfig.forest_type == gbdf::COMPLETE_RANDOM_FOREST) {
            forest = new ClassificationCompleteRF(
                &lconfig.fconfig, 
                vdataset->getNumInstances(), 
                vdataset->getNumFeatures());
        }
        else {
            std::cout << "Error: this type of forest does not exist" << std::endl;
        }
        std::cout << "CC" << std::endl;
        forest->fit(vdataset.get(), vtargets.get());
        forests.push_back(std::shared_ptr<Forest>(forest));
        std::cout << "DD" << std::endl;
    }
    std::cout << "EE" << std::endl;
    grown = true;
}

float* Layer::classify(vdataset_p vdataset) {
    assert(grown);
    size_t n_instances = vdataset->getNumInstances();
    size_t n_classes = lconfig.fconfig.n_classes;
    float* predictions = static_cast<float*>(calloc(n_instances * n_classes, sizeof(float)));
    ClassificationForest* forest_p;
    for (unsigned int i = 0; i < lconfig.n_forests; i++) {
        forest_p = dynamic_cast<ClassificationForest*>(forests.at(i).get());
        float* local_predictions = forest_p->classify(vdataset.get());
        for (unsigned int j = 0; j < n_instances * n_classes; j++) {
            predictions[j] += local_predictions[j];
        }
    }
    return predictions;
}

void Layer::add(layer_p layer) {
    children.push_back(layer);
}

vdataset_p Layer::getVirtualDataset() {
    return this->vdataset;
}

size_t Layer::getNumChildren() {
    return this->children.size();
}

std::ostream& operator<<(std::ostream& os, Layer* const layer) {
    std::cout << "VVVVVV" << std::endl;
    std::cout << layer->getType() << std::endl;
    std::cout << layer->getRequiredMemorySize() << std::endl;
    return os << "Layer type: "   << layer->getType() << std::endl
              << "Virtual size: " << layer->getRequiredMemorySize() << std::endl;
}