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

//' @name Tree
//' @title CART (classification and regression tree)
//' @description CART is an efficient realization of classification and regression tree model in R
//' using Rcpp and Rcpp Armadillo packages. The model is implemented as an S4 class with methods. The model is
//' able to train on an input data and produce predictions based on the learned patterns. Additionally, the model
//' is able to print the tree structure in the matrix form together with decision rules and leaf prediction values.
//' @field new Constructor of the class. This creates an object of the class Tree with pre-defined parameters.
//' Parameters affect how the tree is trained. As such, parameters define the stopping criteria for growing
//' the tree, set the tree type (regression or classification), and identify the type of data that is to be used
//' with the tree (number of features in the data). \itemize{
//' \item Parameter: ident - ID of the tree, must be an integer number
//' \item Parameter: treeType - 0 for classification tree and 1 for regresssion tree
//' \item Parameter: maxNumFeatures - number of features in the data set on which the model is trained
//' \item Parameter: numFeatures - number of features selected at each split
//' \item Parameter: maxDepth - the maximum depth to which the tree is grown
//' \item Parameter: minCount - minimum number of data points for a node to qualify as a leaf node
//' }
//' @field train Train the CART model on the data. This method recursively builds the tree until some pre-specified
//' (in the constructor) stopping criteria is reached. \itemize{
//' \item Parameter: X - The data matrix
//' \item Parameter: Y - matrix of labels
//' @field predict Calculate predictions based on the CART model. This method makes predictions based on the data,
//' using the tree model. The predictions are based on the decision rules created in the training stage. \itemize{
//' \item Parameter: X - The data matrix, based on which predictions are made
//' \item Returns: Y - The vector of predicted values
//' }
//' @field print Print the tree structure of the model. The consecutive rows of matrix represent the nodes. The way
//' the matrix is formed is: first, the node is printed, then the recursive calls are made to print its left and
//' right child nodes respectively. Due to the recursive nature of the print function, the matrix representing the
//' structure of the tree must be handled with care and requires proper attention. \itemize{
//' \item Returns: tr - The data matrix, containing the description of the structure of the tree model. The first
//' column indicates the depth of the node. The second column indicates whether the node is a leaf: 0 - not a leaf
//' 1 - leaf. The third column identifies the feature index based on which the split was performed. The forth column
//' gives the splitting value: all data points which have the feature value less than or equal (<=) to the splitting
//' value are mapped to the left node. Finally, the fifth column indicates the leaf node value if the node is a leaf.
//' }
//' @examples
//' # Classification tree example
//' # Define a data set together with a label
//' X = matrix(1:12, nrow = 6, ncol = 2, byrow = TRUE)
//' Y = c(rep(0, 3), rep(1, 3))
//' # Define a test data set
//' Xtest = matrix(c(-4:-1, 11:14), nrow = 4, ncol = 2, byrow = TRUE)
//' # Define a classification tree object with the following parameters
//' tr = new(Tree, ident = 0, treeType = 0, maxNumFeatures = 2,
//' numFeatures = 2, maxDepth = 4, minCount = 2)
//' # Train the tree
//' tr$train(X, Y)
//' # Make predictions
//' tr$predict(Xtest)
//' # Print tree structure
//' tr$print()
//'
//' # Regression tree example
//' X = matrix(1:12, nrow = 6, ncol = 2, byrow = TRUE)
//' Y = c(0.9, 1.5, 3.2, 11.1, 22.2, 33.3)
//' # Define a test data set
//' Xtest = matrix(c(-4:-1, 11:14), nrow = 4, ncol = 2, byrow = TRUE)
//' # Define a classification tree object with the following parameters
//' tr = new(Tree, ident = 0, treeType = 1, maxNumFeatures = 2,
//' numFeatures = 2, maxDepth = 4, minCount = 2)
//' # Train the tree
//' tr$train(X, Y)
//' # Make predictions
//' tr$predict(Xtest)
//' # Print tree structure
//' tr$print()
class Tree {
public:
  // constructors
  Tree();
  Tree(const int& ident, const int& treeType, const arma::uword& maxNumFeatures,
       const arma::uword& numFeatures, const int& maxDepth, const int& minCount);

  // public methods
  void train(arma::mat& X, arma::colvec& Y);
  arma::colvec predict(const arma::mat& X) const;
  arma::mat print() const;

protected:
  // protected methods
  void printNode(Node* nd, arma::mat& tr) const;
  void buildTree(Node* nd, arma::mat &X, arma::colvec &Y);
  bool stop(const Node* nd, arma::colvec &Y) const;
  bool split(Node* nd, arma::mat &X, arma::colvec &Y);
  void classResult(Node* nd, arma::colvec &Y) const;
  double gini(const std::map<double, int>& classSetLeft, const std::map<double, int>& classSetRight,
              const double& totalSize) const;
protected:
  // protected fields
  int _id;
  int _treeType; // treeType 0: classification tree OR 1: regression tree
  arma::uword _maxNumFeatures; // total features available
  arma::uword _numFeatures; // number of features selected at each split
  int _maxDepth; // maxDepth of tree
  int _minCount; // min count of points for a leaf
  Node* _root;
};

#endif
