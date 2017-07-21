/**
    classification_complete_rf.hpp
    Classification completely-random forests
    
    @author Antoine Passemiers
    @version 1.0 12/04/2017
*/

#ifndef CLASSIFICATION_COMPLETE_RF_HPP_
#define CLASSIFICATION_COMPLETE_RF_HPP_

#include "../misc/sets.hpp"
#include "forest.hpp"
#include "../metrics/classification_metrics.hpp"
#include "../misc/bagging.hpp"


namespace scythe {

class ClassificationCompleteRF : public ClassificationForest {
private:
    std::shared_ptr<ClassificationError> score_metric;
public:
    ClassificationCompleteRF(ForestConfig*, size_t, size_t);
    void init();
    void preprocessDensities(VirtualDataset* dataset);
    void fit(VirtualDataset* dataset, VirtualTargets* targets);
    void fitNewTree(VirtualDataset* dataset, VirtualTargets* targets);
    float* classify(VirtualDataset* dataset);
    ~ClassificationCompleteRF() = default;
};

}

#endif // CLASSIFICATION_COMPLETE_RF_HPP_