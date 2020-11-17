// Include Rcpp system header file (e.g. <>)
#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// Include our definition of the Tree file (e.g. "")
#include "Tree.h"

// Expose (some of) the Student class
RCPP_MODULE(RcppTreeEx){
  Rcpp::class_<Tree>("Tree")
  .default_constructor()
  .method("add", &Tree::add);
}
