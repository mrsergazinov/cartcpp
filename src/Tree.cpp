//  Tree.cpp
// Retrieve the definition of our Tree class
#include "Tree.h"

// Constructors
Node::Node(const arma::uword& d): _depth(d) {}

Tree::Tree() {}

Tree::Tree(const int& ident, const int& treeType, const arma::uword& maxNumFeatures,
           const arma::uword& numFeatures, const int& maxDepth, const int& minCount):
  _id(ident),
  _treeType(treeType),
  _maxNumFeatures(maxNumFeatures),
  _numFeatures(numFeatures),
  _maxDepth(maxDepth),
  _minCount(minCount) {
  // Input checks
  if (_treeType < 0 || _treeType > 1) {
    throw std::range_error("Tree type should be either 0 or 1");
  }
  if (_maxNumFeatures <= 0) {
    throw std::range_error("Max number of features should be > 0");
  }
  if (_numFeatures <= 0) {
    throw std::range_error("Number of features selected at each split should be > 0");
  }
  if (_maxNumFeatures < _numFeatures) {
    throw std::range_error("Number of features selected at each split should be (<=) maximum number of features");
  }
  if (_maxDepth <= 0) {
    throw std::range_error("Max depth > 0");
  }
  if (_minCount <= 0) {
    throw std::range_error("Min count for a leaf node should be > 0");
  }
}

// Methods
void Tree::train(arma::mat &X, arma::colvec &Y) {
  // Input(explicit): data
  // Output: none
  // Process: build a decision tree

  // Input checks
  if (X.n_cols != _maxNumFeatures) {
    throw std::range_error("Dataset should have number of features = max number of features");
  }
  if (X.n_rows <= 0) {
    throw std::range_error("Data set should contain at least 1 data row");
  }
  if (Y.n_elem != X.n_rows) {
    throw std::range_error("Mismatch between dimensions of X and Y");
  }

  _root = new Node(0); // create root node
  _root->_dataPoints = arma::regspace<arma::uvec>(0, 1, X.n_rows - 1); // feed data to the root node
  Tree::buildTree(_root, X, Y); // start building the tree
}

void Tree::buildTree(Node* nd, arma::mat &X, arma::colvec &Y){
  // Input: node, data
  // Output: none
  // Process: check if the node can be split, split the node, continue building the tree

  if (Tree::stop(nd, Y)){
    classResult(nd, Y);
  } else{
    if (split(nd, X, Y)) {
      Tree::buildTree(nd->_left, X, Y);
      Tree::buildTree(nd->_right, X, Y);
    } else {
      classResult(nd, Y);
    }
  }
}

bool Tree::split(Node *nd, arma::mat &X, arma::colvec &Y) {
  // Input: node, data
  // Output: none
  // Process: split the node

  // flag indicating wether at least one splitting value was found
  bool splitted = false;

  // data points sets for next nodes
  arma::uvec leftNodeDatapoints, rightNodeDatapoints;

  //shuffle linear space of column indices
  // select the first numFeatures from the shuffled linear space
  arma::uvec featureSubsetIndex = arma::shuffle(arma::regspace<arma::uvec>(0, 1, _maxNumFeatures - 1));
  featureSubsetIndex = featureSubsetIndex.subvec(0, _numFeatures - 1);

  // Example below explains how we want to sort the values within column
  // 3 1 2 5 - sub-column values
  // 4 5 6 7 - index of sub-column within column
  // 1 2 0 3 - index of sorted sub-column values
  arma::uvec index; // vector to store indices of elements in column as they would appear when sorted

  // create map of class -> count
  std::map<double, int> classSet = {};

  // set the error measures
  double msePrev = std::numeric_limits<double>::infinity(), mseNew = std::numeric_limits<double>::infinity();
  double scoreGiniPrev = 10.0, scoreGiniNew = 11.0;

  // calculate class -> count from the data
  if (_treeType == 0) {
    for (arma::uword i = 0; i < nd->_dataPoints.n_elem; ++i) {
      ++classSet[Y(nd->_dataPoints(i))]; // creating map of label -> count
    }
  }
  std::map<double, int> classSetLeft, classSetRight(classSet);

  for (arma::uword feature : featureSubsetIndex) {
    arma::uvec indFeature = {feature};
    // nd->_dataPoints - rows of dataset (X, Y) correspoding to the node
    // index - vector which describes the sorted order of the elements of the column "feature" among the "dataPoints"
    index = arma::sort_index(X.submat(nd->_dataPoints, indFeature));
    for (arma::uword splitIndex = 0; splitIndex < index.n_elem - 1; ++splitIndex) {
      // type: classification tree
      if (_treeType == 0) {
        ++classSetLeft[Y(nd->_dataPoints(index(splitIndex)))];
        --classSetRight[Y(nd->_dataPoints(index(splitIndex)))];

        if(X(nd->_dataPoints(index(splitIndex)), feature) != X(nd->_dataPoints(index(splitIndex + 1)), feature)) {
          // check that this is NEW split value i.e. different from the previous one
          // this is to avoid unrealistic splitting
          scoreGiniNew = gini(classSetLeft, classSetRight, nd->_dataPoints.n_elem);
        }

        // update node and min Gini score values if the current Gini value is less than the known min
        if (scoreGiniNew < scoreGiniPrev) {
          scoreGiniPrev = scoreGiniNew;
          nd->_featureIndex = feature;
          nd->_splitValue = X(nd->_dataPoints(index(splitIndex)), feature);
          leftNodeDatapoints = nd->_dataPoints(index.head(splitIndex + 1));
          rightNodeDatapoints = nd->_dataPoints(index.tail(index.n_elem - splitIndex - 1));
          splitted = true;
        }
      }
      // type: regression tree
      else {
        // calculate MSE based on the current splitting value
        // make sure that the split is realistic: current splitting value is different from the previous one
        if(X(nd->_dataPoints(index(splitIndex)), feature) != X(nd->_dataPoints(index(splitIndex + 1)), feature)) {

          arma::colvec leftY = Y(index.head(splitIndex + 1));
          arma::colvec rightY = Y(index.tail(index.n_elem - splitIndex - 1));
          arma::colvec leftMse = leftY - arma::mean(leftY);
          arma::colvec rightMse = rightY - arma::mean(rightY);

          mseNew = arma::accu(leftMse % leftMse) * (1 / (double)Y.size()) +
            arma::accu(rightMse % rightMse) * (1 / (double)Y.size());
        }
        // update subsequent nodes and min MSE score accordingly
        if (mseNew < msePrev) {
          msePrev = mseNew;
          nd->_featureIndex = feature;
          nd->_splitValue = X(nd->_dataPoints(index(splitIndex)), feature);
          leftNodeDatapoints = nd->_dataPoints(index.head(splitIndex + 1));
          rightNodeDatapoints = nd->_dataPoints(index.tail(index.n_elem - splitIndex - 1));
          splitted = true;
        }
      }
    }

    // restore the class sets to inital states
    classSetLeft = {};
    classSetRight = classSet;
  }

  if (splitted) {
    nd->_left = new Node(nd->_depth + 1);
    nd->_right = new Node(nd->_depth + 1);
    nd->_left->_dataPoints = leftNodeDatapoints;
    nd->_right->_dataPoints = rightNodeDatapoints;
  }
  return splitted;
}

bool Tree::stop(const Node* nd, arma::colvec &Y) const {
  // Input: node, data
  // Output: boolean indicating whether the node can be split
  // Process: check if the node can be split

  // check that depth is less than maxDepth
  if (nd->_depth >= _maxDepth) {
    return true;
  }
  // check that more than minCount number of points for a leaf
  if (nd->_dataPoints.n_elem <= _minCount) {
    return true;
  }
  // check that the label pool is not homogenuous for a classification tree
  bool first_iter = true;
  double nw = 0.0, prev = 0.0;
  for (const auto& point : nd->_dataPoints) {
    prev = nw;
    nw = Y(point);

    if (first_iter) {
      first_iter = false;
      prev = nw;
    }

    if (nw != prev) {
      return false;
    }
  }
  return true;
}

void Tree::classResult(Node* nd, arma::colvec &Y) const {
  // Input: node, data
  // Output: none
  // Process: calculate leaf value based on the data

  nd->_leaf = true;
  // type: classification tree
  if (_treeType == 0) {
    std::map<double, double> classSet;
    double count = 0;
    for (const auto& point : nd->_dataPoints) {
      classSet[Y(point)] += 1;
      if (classSet[Y(point)] > count) {
        count = classSet[Y(point)];
        nd->_classResult = Y(point);
      }
    }
  }
  // type: regression tree
  else {
    nd->_classResult = arma::mean(Y(nd->_dataPoints));
  }
}

double Tree::gini(const std::map<double, int>& classSetLeft, const std::map<double, int>& classSetRight,
                  const double& totalSize) const {
  double giniVal = 0.0;
  double leftScore = 0.0;
  double leftSize = 0.0;
  double rightScore = 0.0;
  arma::uword rightSize = 0;
  for (auto& classVal : classSetLeft) {
    leftSize += classVal.second;
    leftScore += classVal.second * classVal.second;
  }
  if (leftSize != 0.0) {
    giniVal +=  (1.0 - leftScore / (leftSize * leftSize)) * (leftSize / totalSize);
  }
  for (auto& classVal : classSetRight) {
    rightSize += classVal.second;
    rightScore += classVal.second * classVal.second;
  }
  if (rightSize != 0.0) {
    giniVal += (1.0 - rightScore / (rightSize * rightSize)) * (rightSize / totalSize);
  }
  return giniVal;
}

arma::colvec Tree::predict(const arma::mat& X) const {
  // Input checks
  if (X.n_rows <= 0) {
    throw std::range_error("Data set should contain at least 1 data row");
  }
  if (X.n_cols != _maxNumFeatures) {
    throw std::range_error("Data set should have the number of features = max number of features");
  }

  // starting at the root
  Node* nd = _root;
  // vector to store predicted values
  arma::colvec Ypred(X.n_rows);

  // start at the root node and conseqeuntly go down the tree until a leaf node is reached
  // repeat for each data row
  for (arma::uword row = 0; row < X.n_rows; ++row) {
    while (!nd->_leaf) {
      if (X(row, nd->_featureIndex) <= nd->_splitValue) {
        nd = nd->_left;
      } else {
        nd = nd->_right;
      }
    }
    Ypred(row) = nd->_classResult;
    nd = _root;
  }
  return Ypred;
}

arma::mat Tree::print() const {
  // print tree recursively starting from a root
  // matrix has
  // 1st column: depth
  // 2nd column: leaf or not (1 - leaf, 0 -  not a leaf)
  // 3rd column: feature index
  // 4th column: split value
  // 5th column: class result if leaf
  // first the left node row is added, then right node
  arma::mat tr(0, 5);
  Node* nd = _root;
  printNode(nd, tr);

  return tr;
}

void Tree::printNode(Node* nd, arma::mat& tr) const {
  if (nd->_leaf) {
    tr.insert_rows(tr.n_rows, 1);
    arma::rowvec vec = {(double)nd->_depth, 1, 0, 0, nd->_classResult};
    tr.row(tr.n_rows - 1) = vec;
  } else {
    tr.insert_rows(tr.n_rows, 1);
    arma::rowvec vec = {(double)nd->_depth, 0, (double)nd->_featureIndex,  nd->_splitValue, 0};
    tr.row(tr.n_rows - 1) = vec;
    printNode(nd->_left, tr);
    printNode(nd->_right, tr);
  }
}
