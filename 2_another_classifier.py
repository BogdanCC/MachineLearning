# Steps for this program
    # 1. Import data set
    # 2. Train classifier
    # 3. Predict label for new data

# Import the iris data set from sklean ( this is a table with 150 examples of 3 types of flower - 50 examples each)
# The features are : Sepal length, Sepal width, Petal length, Petal width
# The labels are (the flower species) : Setosa, Versicolor, Virginica
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
iris_data = load_iris()
# target_names returns the metadata for the labels : Setosa, Versicolor, Virginica
print("Label names: ", iris_data.target_names)
# feature_names returns the metadata for the features : Sepal length, Sepal width, Petal length, Petal width
print("Feature names : ", iris_data.feature_names)
# data[] returns the features data
print(iris_data.data[0])
# target[] returns the label data
print(iris_data.target[0])
# using the for loop to show all data
for i in range(len(iris_data.data)):
    print("Label : {}; Features : {}".format(iris_data.target[i], iris_data.data[i]))

# Now we can train the classifier but first we need to remove at least one flower example so we can feed it in afterwards and see if it guesses correctly
# Aka training data vs testing data
# We will remove 4 setos(indeces 0, 1, 3, 10), 2 versicolor (indeces 50, 51), 2 virginica(indeces 100, 121)
import numpy as np
data_to_remove = [[0, 1, 50, 10, 51, 3, 100, 121]]

# training data (axis 0 simply means we're deleting features rows, not columns)
training_data = np.delete(iris_data.data, data_to_remove, axis = 0)
training_target = np.delete(iris_data.target, data_to_remove)

# testing data
test_data = iris_data.data[data_to_remove]
test_target = iris_data.target[data_to_remove]

# 2. Training classifier - Let's use 2 different learning algorithms
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_target)
clfG = GaussianNB()
clfG.fit(training_data, training_target)

# 3. Making predictions
print("Correct labels for test data :\n", test_target)
print("(Dec. Tree) Program predicts :\n", clf.predict(test_data))
print("(Naive Bayes) Program predicts : \n", clfG.predict(test_data))
