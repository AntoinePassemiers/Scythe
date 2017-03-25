#ifndef FOREST_HPP_
#define FOREST_HPP_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../id3.hpp"


class Forest {
private:
	size_t n_trees;
	std::vector<struct Tree*> trees;

public:
	Forest() : n_trees(0), trees() {}
};

#endif // FOREST_H_