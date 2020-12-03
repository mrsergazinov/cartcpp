#ifndef Tree_H
#define Tree_H

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

class Node {
public:
  // methods
  Node();
  Node(const int& d);
  Node(const Node& other);
  // ~Node();

public:
  // fields
  Node* _left;
  Node* _right;
  int _depth;
  arma::mat* _dataSet;
  arma::colvec* _label;
  int _featureIndex;
  double _splitValue;
  double _classResult;
};

class Tree {
public:
  // constructors
  Tree();
  Tree(const int& ident, const int& treeType, const arma::uword& numFeatures,
       const int& maxDepth, const int& minCount);

  // public methods
  void train(arma::mat& X, arma::colvec& Y);
  arma::colvec predict(const arma::mat& X) const;

protected:
  // protected methods
  void buildTree(Node* nd);
  bool stop(const Node* nd) const;
  bool split(Node* nd);
  void classResult(Node* nd, const int treeType);
  double gini(const arma::mat& X, const arma::colvec& Y, const arma::uword& featureId, const double& featureVal) const;
  double mse(const arma::mat& X, const arma::colvec& Y, const arma::uword& featureId, const double& featureVal) const;

protected:
  // protected fields
  arma::uword _numFeatures; // ratio of features selected at each split
  Node* _root;
  int _id;
  int _treeType; // treeType 0: classification tree OR 1: regression tree
  int _maxDepth; // maxDepth of tree
  int _minCount; // min count of points for a leaf
};

#endif
