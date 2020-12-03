//  Tree.cpp
// Retrieve the definition of our Tree class
#include "Tree.h"

// Constructors
Node::Node(const int& d): _depth(d) {}

Tree::Tree() {}

Tree::Tree(const int& ident, const int& treeType, const arma::uword& numFeatures,
           const int& maxDepth, const int& minCount):
  _id(ident),
  _treeType(treeType),
  _numFeatures(numFeatures),
  _maxDepth(maxDepth),
  _minCount(minCount) {}

// Methods
arma::mat Tree::add(const arma::mat& A, const arma::mat& B) {
  return (A + B);
}
