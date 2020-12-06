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
  _minCount(minCount) {}

// Methods
void Tree::train(arma::mat &X, arma::colvec &Y) {
  // Input(explicit): data
  // Output: none
  // Process: build a decision tree

  //TODO: make compatibility checks

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
    split(nd, X, Y);
    Tree::buildTree(nd->_left, X, Y);
    Tree::buildTree(nd->_right, X, Y);
  }
}

void Tree::split(Node *nd, arma::mat &X, arma::colvec &Y) {
  // Input: node, data
  // Output: none
  // Process: split the node

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

  std::cout << "Arrived at a loop" << std::endl;
  for (arma::uword feature : featureSubsetIndex) {
    arma::uvec indFeature = {feature};
    // nd->_dataPoints - rows of dataset (X, Y) correspoding to the node
    // index - vector which describes the sorted order of the elements of the column "feature" among the "dataPoints"
    index = arma::sort_index(X.submat(nd->_dataPoints, indFeature));
    std::cout << "Iteration for feature: " << feature << std::endl;
    std::cout << "Sorted index array:" << std::endl;
    std::cout << index << std::endl;
    for (arma::uword splitIndex = 0; splitIndex < index.n_elem - 1; ++splitIndex) {
      std::cout << "index(splitIndex): " << index(splitIndex) << std::endl;
      std::cout << "nd->_dataPoints(index(splitIndex))): " << nd->_dataPoints(index(splitIndex)) << std::endl;

      // type: classification tree
      if (_treeType == 0) {
        ++classSetLeft[Y(nd->_dataPoints(index(splitIndex)))];
        --classSetRight[Y(nd->_dataPoints(index(splitIndex)))];

        if(X(nd->_dataPoints(index(splitIndex)), feature) != X(nd->_dataPoints(index(splitIndex + 1)), feature)) {
          // check that this is NEW split value i.e. different from the previous one
          // this is to avoid unrealistic splitting
          std::cout << "New gini score calculated: different split value" << std::endl;
          scoreGiniNew = gini(classSetLeft, classSetRight, nd->_dataPoints.n_elem);
          std::cout << "scoreGiniNew: " << scoreGiniNew << std::endl;
        }

        // update node and min Gini score values if the current Gini value is less than the known min
        if (scoreGiniNew < scoreGiniPrev) {
          std::cout << "Previous gini score: " << scoreGiniPrev << std::endl;
          std::cout << "The splitting value is updated" << std::endl;
          scoreGiniPrev = scoreGiniNew;
          nd->_featureIndex = feature;
          nd->_splitValue = X(nd->_dataPoints(index(splitIndex)), feature);
          std::cout << "index for left node: " <<index.head(splitIndex + 1) << std::endl;
          leftNodeDatapoints = nd->_dataPoints(index.head(splitIndex + 1));
          rightNodeDatapoints = nd->_dataPoints(index.tail(index.n_elem - splitIndex - 1));
          std::cout << "Data points for left node are: " << std::endl << leftNodeDatapoints << std::endl;
          std::cout << "Data points for right node are: " << std::endl << rightNodeDatapoints << std::endl;
        }
      }
      // type: regression tree
      else {
        // calculate MSE based on the current splitting value
        // make sure that the split is realistic: current splitting value is different from the previous one
        if(X(nd->_dataPoints(index(splitIndex)), feature) != X(nd->_dataPoints(index(splitIndex + 1)), feature)) {
          std::cout << "MSE updated" << std::endl;
          arma::colvec leftY = Y(index.head(splitIndex + 1));
          std::cout << "left Y: " << std::endl << leftY << std::endl;
          arma::colvec rightY = Y(index.tail(index.n_elem - splitIndex - 1));
          std::cout << "right Y: " << std::endl << rightY << std::endl;
          arma::colvec leftMse = leftY - arma::mean(leftY);
          arma::colvec rightMse = rightY - arma::mean(rightY);
          std::cout << "left mse Y: " << std::endl << leftMse << std::endl;
          std::cout << "right mse Y: " << std::endl << rightMse << std::endl;

          mseNew = arma::accu(leftMse % leftMse) * (1 / (double)Y.size()) +
            arma::accu(rightMse % rightMse) * (1 / (double)Y.size());

          std::cout << "new Mse: " << mseNew << std::endl;
        }
        // update subsequent nodes and min MSE score accordingly
        if (mseNew < msePrev) {
          std::cout << "prev Mse: " << msePrev << std::endl;
          msePrev = mseNew;
          nd->_featureIndex = feature;
          nd->_splitValue = X(nd->_dataPoints(index(splitIndex)), feature);
          leftNodeDatapoints = nd->_dataPoints(index.head(splitIndex + 1));
          rightNodeDatapoints = nd->_dataPoints(index.tail(index.n_elem - splitIndex - 1));
          std::cout << "index for left node: " << std::endl << index.head(splitIndex + 1) << std::endl;
          std::cout << "Data points for left node are: " << std::endl << leftNodeDatapoints << std::endl;
          std::cout << "Data points for right node are: " << std::endl << rightNodeDatapoints << std::endl;
        }
      }
    }

    // restore the class sets to inital states
    classSetLeft = {};
    classSetRight = classSet;
  }

  nd->_left = new Node(nd->_depth + 1);
  nd->_right = new Node(nd->_depth + 1);
  nd->_left->_dataPoints = leftNodeDatapoints;
  nd->_right->_dataPoints = rightNodeDatapoints;
}

bool Tree::stop(const Node* nd, arma::colvec &Y) const {
  // Input: node, data
  // Output: boolean indicating whether the node can be split
  // Process: check if the node can be split

  // check that depth is less than maxDepth
  if (nd->_depth > _maxDepth) {
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
  std::cout << "leftSize: " << leftSize << std::endl;
  std::cout << "leftScore: " << leftScore << std::endl;
  if (leftSize != 0.0) {
    giniVal +=  (1.0 - leftScore / (leftSize * leftSize)) * (leftSize / totalSize);
  }
  for (auto& classVal : classSetRight) {
    rightSize += classVal.second;
    rightScore += classVal.second * classVal.second;
  }
  std::cout << "rightSize: " << rightSize << std::endl;
  std::cout << "rightScore: " << rightScore << std::endl;
  if (rightSize != 0.0) {
    giniVal += (1.0 - rightScore / (rightSize * rightSize)) * (rightSize / totalSize);
  }
  return giniVal;
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
