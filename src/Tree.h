#ifndef Tree_H
#define Tree_H

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]


class Node {
public:
  Node();
  Node(const int& d);
  Node(const Node& other);
  // ~Node();

protected:
  Node* _left;
  Node* _right;
  int _depth;
  int _featureIndex;
  double _splitValue;
  double _classResult;
  arma::mat* _dataSet;
};

class Tree {
public:
  Tree();
  Tree(const int& id, const int& treeType, const double& testRatio,
       const int& maxDepth, const int& minCount);

  void train(const arma::mat& X);
  arma::colvec predict(const arma::mat& X) const;
  arma::mat add(const arma::mat& A, const arma::mat& B);

protected:
  void buildTree(Node* nd);
  bool stop(const Node* nd) const;
  bool split(Node* nd);
  void classResult(Node* nd);
  double gini(const arma::mat& X, const arma::uword& featureId, const double& featureVal) const;
  double mse(const arma::mat& X, const arma::uword& featureId, const double& featureVal) const;

protected:
  int _id;
  Node* _root;
  int _treeType;   // treeType 0: classification tree OR 1: regression tree
  double _testRatio; // ratio of features selected at each split
  int _maxDepth; // maxDepth of tree
  int _minCount; // min count of points for a leaf
};

#endif
