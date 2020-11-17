//  Tree.cpp
// Retrieve the definition of our Tree class
#include "Tree.h"

// Constructors ----
Tree::Tree() {}


arma::mat Tree::add(const arma::mat& A, const arma::mat& B) {
  return (A + B);
}
