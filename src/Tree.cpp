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
  _root->_dataSet = &X; // feed data to the root node
  _root->_label = &Y;
  Tree::buildTree(_root); // start building the tree
}

void Tree::buildTree(Node* node){

  if (Tree::stop(node)){
    classResult(node, _treeType);
  } else{
    split(node);
    Tree::buildTree(node->_left);
    Tree::buildTree(node->_right);
  }
}
