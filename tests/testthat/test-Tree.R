test_that("Input checks work", {
  # incorrect tree type
  expect_error(new(Tree, ident = 0, treeType = 2,
                   maxNumFeatures = 4, numFeatures = 2,
                   maxDepth = 10, minCount = 2))
  expect_error(new(Tree, ident = 0, treeType = -1,
                   maxNumFeatures = 4, numFeatures = 2,
                   maxDepth = 10, minCount = 2))
  expect_error(new(Tree, ident = 0, treeType = 3,
                   maxNumFeatures = 4, numFeatures = 2,
                   maxDepth = 10, minCount = 2))

  # incorrect number of features specified
  expect_error(new(Tree, ident = 0, treeType = 0,
                   maxNumFeatures = 0, numFeatures = 2,
                   maxDepth = 10, minCount = 2))
  expect_error(new(Tree, ident = 0, treeType = 0,
                   maxNumFeatures = 2, numFeatures = 0,
                   maxDepth = 10, minCount = 2))
  expect_error(new(Tree, ident = 0, treeType = 0,
                   maxNumFeatures = 2, numFeatures = -1,
                   maxDepth = 10, minCount = 2))
  expect_error(new(Tree, ident = 0, treeType = 0,
                   maxNumFeatures = 4, numFeatures = 6,
                   maxDepth = 10, minCount = 2))

  # training set contains more features than specified by parameter maxNumFeatures
  tr = new(Tree, ident = 0, treeType = 0,
      maxNumFeatures = 4, numFeatures = 2,
      maxDepth = 10, minCount = 2)
  X = matrix(1:40, nrow = 8, ncol = 5)
  Y = 1:8
  expect_error(tr$train(X, Y))

  # training labels have different dimensions than training data set
  tr = new(Tree, ident = 0, treeType = 0,
           maxNumFeatures = 5, numFeatures = 2,
           maxDepth = 10, minCount = 2)
  X = matrix(1:40, nrow = 8, ncol = 5)
  Y = 1:10
  expect_error(tr$train(X, Y))
  Y = 1:5
  expect_error(tr$train(X, Y))

  # test data has different number of features than the training data
  tr = new(Tree, ident = 0, treeType = 0,
           maxNumFeatures = 5, numFeatures = 2,
           maxDepth = 10, minCount = 2)
  X = matrix(1:40, nrow = 8, ncol = 5)
  Y = 1:8
  tr$train(X, Y)
  Xtest = matrix(1:40, nrow = 5, ncol = 8)
  expect_error(tr$predict(Xtest))
  Xtest = matrix(1:40, nrow = 10, ncol = 4)
  expect_error(tr$predict(Xtest))
})

test_that("Input checks work", {
  # toy example 1: classification
  # create a separable data set with 1 feature
  X = matrix(c(0, 1, 2, 10, 20, 30, 5, 6, 15), ncol = 1)
  Y = c(0, 0, 0, 1, 1, 1, 0, 0, 1)
  # define Tree object
  tr = new(Tree, ident = 0, treeType = 0,
           maxNumFeatures = 1, numFeatures = 1,
           maxDepth = 100, minCount = 2)
  # fit the model
  tr$train(X, Y)
  # create a test set
  Xtest = matrix(c(-1, -2, 33, 21))
  expect_equal(as.vector(tr$predict(Xtest)), c(0, 0, 1, 1))

  # toy example 2: classification
  # create a separable data set with 2 features
  X = matrix(c(c(0, 1, 2, 10, 20, 30, 5, 6, 15),
               c(30, 32.2, 20, 57, 90, 56.6, 23.3, 43, 60.66)), ncol = 2)
  Y = c(0, 0, 0, 1, 1, 1, 0, 0, 1)
  # define Tree object
  tr = new(Tree, ident = 0, treeType = 0,
           maxNumFeatures = 2, numFeatures = 1,
           maxDepth = 100, minCount = 2)
  # fit the model
  tr$train(X, Y)
  # create a test set
  Xtest = matrix(c(c(-1, -2, 33, 21),
                 c(0, 10, 76, 87.7)), ncol = 2)
  expect_equal(as.vector(tr$predict(Xtest)), c(0, 0, 1, 1))


})

