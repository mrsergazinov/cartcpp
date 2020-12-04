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
  arma::uword _beginRow;
  arma::uword _endRow;
  int _featureIndex;
  double _splitValue;
  double _classResult;
};

class Tree {
public:
  // constructors
  Tree();
  Tree(const int& ident, const int& treeType, const arma::uword& maxNumFeatures,
       const arma::uword& numFeatures, const int& maxDepth, const int& minCount);

  // public methods
  void train(arma::mat& X, arma::colvec& Y);
  arma::colvec predict(const arma::mat& X) const;

protected:
  // protected methods
  void buildTree(Node* nd, arma::mat &X, arma::colvec &Y);
  bool stop(const Node* nd, arma::mat &X, arma::colvec &Y) const;
  void split(Node* nd, arma::mat &X, arma::colvec &Y);
  void classResult(Node* nd, arma::mat &X, arma::colvec &Y) const;
  double gini(const arma::mat& X, const arma::colvec& Y, const arma::uword& featureId, const double& featureVal) const;
  double mse(const arma::mat& X, const arma::colvec& Y, const arma::uword& featureId, const double& featureVal) const;

protected:
  // protected fields
  arma::uword _maxNumFeatures; // ratio of features selected at each split
  arma::uword _numFeatures; // ratio of features selected at each split
  Node* _root;
  int _id;
  int _treeType; // treeType 0: classification tree OR 1: regression tree
  int _maxDepth; // maxDepth of tree
  int _minCount; // min count of points for a leaf
};

#endif
