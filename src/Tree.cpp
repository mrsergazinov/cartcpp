//  Tree.cpp
// Retrieve the definition of our Tree class
#include "Tree.h"

// Constructors
Node::Node(const int& d): _depth(d) {}

Tree::Tree() {}

Tree::Tree(const int& ident, const int& treeType, const arma::uword& maxNumFeatures,
           const arma::uword& numFeatures, const int& maxDepth, const int& minCount):
  _id(ident),
  _treeType(treeType),
  _maxNumFeatures(maxNumFeatures),
  _numFeatures(numFeatures),
  _maxDepth(maxDepth),
  _minCount(minCount) {}

// Methods
void Tree::train(arma::mat &X, arma::colvec &Y) {
  // Input(explicit): data
  // Output: none
  // Process: build a decision tree

  //TODO: make compatibility checks

  _root = new Node(0); // create root node
  _root->_beginRow = 0; // feed data to the root node
  _root->_endRow = X.n_rows - 1;
  Tree::buildTree(_root, X, Y); // start building the tree
}

void Tree::buildTree(Node* nd, arma::mat &X, arma::colvec &Y){
  // Input: node, data
  // Output: none
  // Process: check if the node can be split, split the node, continue building the tree

  if (Tree::stop(nd, X, Y)){
    classResult(nd, X, Y);
  } else{
    split(nd, X, Y);
    Tree::buildTree(nd->_left, X, Y);
    Tree::buildTree(nd->_right, X, Y);
  }
}

void Tree::split(Node *nd, arma::mat &X, arma::colvec &Y) {
  // Input: node, data
  // Output: none
  // Process: split the node
  arma::colvec subsetIndex = arma::shuffle(arma::linspace(0, X.n_cols - 1)); //shuffle linear space of column indices
  subsetIndex = subsetIndex.subvec(0, _numFeatures - 1); // select the first numFeatures from the shuffled linear space

  for (arma::uword feature : subsetIndex) {

  }

  nd->_left = new Node(nd->_depth + 1);
  nd->_left = new Node(nd->_depth + 1);

}

bool Tree::stop(const Node* nd, arma::mat &X, arma::colvec &Y) const {
  // Input: node, data
  // Output: boolean indicating whether the node can be split
  // Process: check if the node can be split

  // TODO: fill-in the code
  // check that depth is less than maxDepth
  // check that more than minCount number of points for a leaf
  // check that the label pool is not homogenuous for a classification tree
  return false;
}

void Tree::classResult(Node* nd, arma::mat &X, arma::colvec &Y) const {
  // Input: node, data
  // Output: none
  // Process: calculate leaf value based on the data

  //TODO: fill-in the code
  nd->_classResult = 1;
}

void Tree::print() const {
  std::cout << "Is root a leaf: " << _root->_leaf << std::endl;
  Tree::printNode(_root);
}

int Tree::printNode(Node* nd) const {
  if (nd == nullptr) {
    return 0;
  }
  if (nd->_leaf) {
    std::cout << "Leaf node with result: " << nd->_classResult << std::endl << std::endl;
  } else {
    std::cout << "Not a leaf node" << std::endl;
    std::cout << "Feature index: " << std::endl << nd->_featureIndex << std::endl;
    std::cout << "Split value: " << std::endl << nd->_splitValue << std::endl;
    std::cout << "Left set: " << std::endl << nd->_left->_dataPoints << std::endl;
    std::cout << "Right set: " << std::endl << nd->_right->_dataPoints << std::endl << std::endl;
  }
  Tree::printNode(nd->_left);
  Tree::printNode(nd->_right);
  return 0;
}
