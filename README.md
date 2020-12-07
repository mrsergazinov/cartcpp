## Cartcpp package

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






