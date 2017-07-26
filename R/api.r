rm(list = ls())

install.packages('Rcpp')
install.packages("devtools")

library(Rcpp)
library(devtools)

Rcpp.package.skeleton("scythe")

"TODO : load the lib and wrap the C functions"
PATH <- "C:/Users/Xanto183/git/Scythe/src/scythe.lib";
dyn.load(PATH);
