---
title: "cartcpp package"
output: html_document
bibliography: bibliography.bib
---

### Description
A decision tree is a predictive model, which uses a tree structure to go from observations to conclusions.
Decision trees are one of the most ubiquitous supervised learning tools. At each node of the tree, some
decision rule is set so that to funnel the observation to the most appropriate leaf. Trees with discrete labels
are known as classification trees, whereas trees with continuous labels are known as regression trees. A typical
way to fit or train the tree is the greedy algorithm approach, called top-down induction. Under the top-down
induction, each node of the tree is recursively split until a split no longer adds information or a required
depth is reached.

The term CART (classification and regression tree) was introduced by @breiman1984classification to refer to
both types of trees. A single CART model can be used off-the-shelf since it is invariant under feature
transformations and scaling, robust to the inclusion of irrelevant features, and produces interpretable results.
Nevertheless, a single CART is rarely accurate as it frequently suffers from overfitting. Hence, @breiman2001random has proposed a random forest algorithm, which essentially averages the results of multiple trees trained on different parts of the training data. Unlike the traditional bagging approach, where an ensemble of models is trained on different subsets of the data, random forests are trained so that at each node split is selected based on a random subset of features. This technique allows reducing the correlation between the individual trees greatly. A peculiar characteristic of random forests is their resistance to overfitting. Indeed, it can be shown that the accuracy of predictions increases nearly monotonically with the complexity (size) of the random forest.

### Functionality

This package offers an efficient implementation of CART model with Rcpp and  Rcpp Armadillo data
structures and algorithms. The CART model is able to train on the data and output predictions.
Additionally, the model is able to select a random subset of features at each split. Finally, the tree
model also has options for multiple stopping criteria such as maximum depth or minimum samples
required to be a leaf node.

The package also implements a random forest algorithm. The random forest implementation has options for the 
number of estimators and the parameters for each estimator.

### References


