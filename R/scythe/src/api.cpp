#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include "scythe.hpp"

// [[Rcpp::export]]
double foo() {
	double a;
	std::cout << "Hello world !" << std::endl;
	return a;
}