## Cartcpp package

- [Description](#description)
- [Functionality](#functionality)
- [Installation](#installation)
- [Usage](#usage)
    - [1. Classification](#classification)
    - [2. Regression](#regression)
- [Q&A](#Q&A)


### Description
A decision tree is a predictive model, which uses a tree structure to go from observations to conclusions.
Decision trees are one of the most ubiquitous supervised learning tools. At each node of the tree, some
decision rule is set so that to funnel the observation to the most appropriate leaf. Trees with discrete labels
are known as classification trees, whereas trees with continuous labels are known as regression trees. A typical
way to fit or train the tree is the greedy algorithm approach, called top-down induction. Under the top-down
induction, each node of the tree is recursively split until a split no longer adds information or a required
depth is reached.

The term CART (classification and regression tree) was introduced by Breiman, 1984 to refer to
both types of trees. A single CART model can be used off-the-shelf since it is invariant under feature
transformations and scaling, robust to the inclusion of irrelevant features, and produces interpretable results. CART freqruntly serves as a workhorse for other advanced models such as random forest and gradient boosted trees. These methods can be built rather quickly once the CART model has been implemented. 

### Functionality

This package offers an efficient implementation of CART model with Rcpp and  Rcpp Armadillo data
structures and algorithms. The CART model is able to train on the data and output predictions.
Additionally, the model is able to select a random subset of features at each split. Finally, the tree
model also has options for multiple stopping criteria such as maximum depth or minimum samples
required to be a leaf node. Furthermore, the model is able to print its strucutre together with the decision rules in the matrix form, which allows for conveiniet examination. 


### Installation

Step 1. Download necessary compiler tools:
Install [Rtools for Windows](https://cran.r-project.org/bin/windows/Rtools/) and [GFortran for Mac OS](https://gcc.gnu.org/wiki/GFortranBinariesMacOS).

Step 2. Install package devtools in R:

```R
install.packages("devtools")
```

Step 3. Build package using devtools:

```R
devtools::install_github("mrsergazinov/cartcpp")
```

Step 4. Load the package in R using:

```R
library(cartcpp)
```

### Usage

To learn more about the Tree class methods, their parameters, and outputs, consider running:
```R
?Tree
```
Also, to explore the documentation for any particular method, one could also run:
```R
?`Tree$train`
```

#### Classification
Here is a simple example demonstrating how to use CART model class for the classification purposes (the data here is gnerated by hand):
```R
# Example 1: Create a data set with labels, only 2 features
X = matrix(c(rnorm(100, 0, 2), rnorm(100, 20, 3)), nrow = 100, ncol = 2)
Y = c(rep(0, 30), rep(1, 30), rep(2, 40))
# Define a tree object
tr = new(Tree, ident = 123, treeType = 0, maxNumFeatures = 2,
numFeatures = 2, maxDepth = 100, minCount = 2)
# Fit the tree
tr$train(X, Y)
# Print the tree structure
tr$print()
# Make predictions
Xtest = matrix(c(rnorm(10, 0, 2), rnorm(10, 20, 3)), nrow = 10, ncol = 2)
tr$predict(Xtest)

# Example 2: Create a data set with labels, 20 features
X = matrix(c(rnorm(200, 0, 2), rnorm(200, 20, 3)), nrow = 20, ncol = 20)
Y = c(rep(0, 5), rep(1, 7), rep(2, 8))
# Define a tree object
tr = new(Tree, ident = 123, treeType = 0, maxNumFeatures = 20,
numFeatures = 5, maxDepth = 100, minCount = 2)
# Fit the tree
tr$train(X, Y)
# Print the tree structure
tr$print()
# Make predictions
Xtest = matrix(c(rnorm(40, 0, 2), rnorm(40, 20, 3)), nrow = 4, ncol = 20)
tr$predict(Xtest)
```

#### Regression
Another way to use CART is for regression. The example below demonstrated how CART model class could be used for the regression purposes:
```R
# Example 1: Create a data set with labels with 4 features
X = matrix(c(rnorm(100, 0, 2), rnorm(100, 20, 3)), nrow = 50, ncol = 4)
Y = c(rnorm(25, 0, 3), rnorm(25, 10, 1))
# Define a tree object
tr = new(Tree, ident = 123, treeType = 1, maxNumFeatures = 4,
numFeatures = 3, maxDepth = 100, minCount = 2)
# Fit the tree
tr$train(X, Y)
# Print the tree structure
tr$print()
# Make predictions
Xtest = matrix(c(rnorm(10, 0, 2), rnorm(10, 20, 3)), nrow = 5, ncol = 4)
tr$predict(Xtest)
```

### Q&A
Please, direct your questions and concerns to my email address, which can be found in my Github account.


