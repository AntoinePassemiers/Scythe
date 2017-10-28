/**
    pruning.hpp
    Post-pruning on classification and regression trees

    @author Antoine Passemiers
    @version 1.0 09/08/2017
*/

#ifndef PRUNING_HPP_
#define PRUNING_HPP_

#include "../tree/cart.hpp"


namespace scythe {


struct NodeLevel {
    Node* owner;
    size_t level;
};

struct Cut {
    Node* leaf;
    Node* left;
    Node* right;  
};


class Scythe {
private:
	std::vector<Tree*> trees;
    std::vector<std::vector<Cut>> prunings;
    size_t n_prunings;
    bool delete_branches;
public:
	Scythe();

	void add(Tree* tree) { trees.push_back(tree); }
	size_t cut(Node* node);
	int prune(size_t max_depth);
    void restore(int pruning_id);
    void prune(int pruning_id);
};

} // namespace

#endif // PRUNING_HPP_