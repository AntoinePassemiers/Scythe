/**
    pruning.hpp
    Post-pruning on classification and regression trees

    @author Antoine Passemiers
    @version 1.0 09/08/2017
*/

#include "pruning.hpp"


namespace scythe {

Scythe::Scythe() : trees(), prunings(), n_prunings(0), delete_branches(false) {}

size_t Scythe::cut(Node* root) {
    size_t n_removed_nodes = 0;
    std::queue<Node*> queue;
    queue.push(root->left_child);
    queue.push(root->right_child);
    while (!queue.empty()) {
        Node* node = queue.front(); queue.pop();
        if (node->left_child != nullptr) { queue.push(node->left_child); }
        if (node->right_child != nullptr) { queue.push(node->right_child); }
        delete node; n_removed_nodes++;
    }
    root->left_child = nullptr;
    root->right_child = nullptr;
    return n_removed_nodes;
}

int Scythe::prune(size_t max_depth) {
    std::vector<Cut> slicings;
    for (Tree* tree : trees) {
        std::queue<NodeLevel> queue;
        NodeLevel current_node_space;
        current_node_space.owner = tree->root;
        current_node_space.level = 1;
        queue.push(current_node_space);
        while (!queue.empty()) {
            NodeLevel current_node_space = queue.front(); queue.pop();
            Node* current_node = current_node_space.owner;
            if (current_node->left_child != nullptr) {
                if (current_node_space.level >= max_depth) {
                    // tree->n_nodes -= cut(current_node);
                    Cut slicing;
                    slicing.leaf  = current_node;
                    slicing.left  = current_node->left_child;
                    slicing.right = current_node->right_child;
                    slicings.push_back(slicing);
                    current_node->left_child = nullptr;
                    current_node->right_child = nullptr;
                }
                else {
                    NodeLevel left_space;
                    left_space.owner = current_node->left_child;
                    left_space.level = current_node_space.level + 1;
                    queue.push(left_space);
                    NodeLevel right_space;
                    right_space.owner = current_node->left_child;
                    right_space.level = current_node_space.level + 1;
                    queue.push(right_space);
                }
            }
        }
    }
    prunings.push_back(slicings);
    return n_prunings++;
}

void Scythe::restore(int pruning_id) {
    for (Cut slicing : prunings.at(pruning_id)) {
        slicing.leaf->left_child  = slicing.left;
        slicing.leaf->right_child = slicing.right;
    }
}

void Scythe::prune(int pruning_id) {
    for (Cut slicing : prunings.at(pruning_id)) {
        slicing.leaf->left_child  = nullptr;
        slicing.leaf->right_child = nullptr;
    }
}

} // namespace