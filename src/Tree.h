#ifndef Tree_H
#define Tree_H
#include <map>
#include <iostream>
#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

class Node {
public:
  // methods
  Node();
  Node(const arma::uword& d);
  Node(const Node& other);
  // ~Node();

public:
  // fields
  Node* _left;
  Node* _right;
  arma::uword _depth;
  arma::uvec _dataPoints;
  arma::uword _featureIndex;
  bool _leaf = false;
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
  void print() const;

protected:
  // protected methods
  int printNode(Node* nd) const;
  void buildTree(Node* nd, arma::mat &X, arma::colvec &Y);
  bool stop(const Node* nd, arma::colvec &Y) const;
  bool split(Node* nd, arma::mat &X, arma::colvec &Y);
  void classResult(Node* nd, arma::colvec &Y) const;
  double gini(const std::map<double, int>& classSetLeft, const std::map<double, int>& classSetRight,
              const double& totalSize) const;
protected:
  // protected fields
  arma::uword _maxNumFeatures; // total features available
  arma::uword _numFeatures; // number of features selected at each split
  Node* _root;
  int _id;
  int _treeType; // treeType 0: classification tree OR 1: regression tree
  int _maxDepth; // maxDepth of tree
  int _minCount; // min count of points for a leaf
};

#endif
