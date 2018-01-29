# Our classifier : scrappy version of the k-nearest neighbors
import random
# We'll use the euclidean formula to calculate the distance between features
from scipy.spatial import distance
# a - list of training data (features) ; b - list of testing data (features)
def euc(a,b):
    return distance.euclidean(a,b)

# Defining our classifier class
class ScrappyKNN():
    # As we saw in the pre-built classifiers, we need to have 2 main functions
    # Fit and Predict
    # Fit will simply assign the data we're working with
    def fit(self, features_train, labels_train):
        self.features_train = features_train
        self.labels_train = labels_train
    # Predict will loop through the testing features and find the distance between them
    # and the nearest neighbors (training features). It will return the label using that distance
    def predict(self, features_test):
        # store the labels in the predictions list
        predictions = []
        for row in features_test:
            label = self.closest(row)
            predictions.append(label)
        # return the predictions list
        return predictions
    # Defining classifier algorithm, it takes a row of test features as argument
    def closest(self, row):
        # initialise best distance and index
        best_dist = euc(row, self.features_train[0])
        best_index = 0
        # loop through features_train list and compare our test row with every train row
        # if it finds a distance smaller than the initial distance, it will set it as the best_dist
        # save the index in best_index to return it as label
        for i in range(1, len(self.features_train)):
            dist = euc(row, self.features_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.labels_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size = .5)

# from sklearn.neighbors import KNeighborsClassifier

my_classifier = ScrappyKNN()
my_classifier.fit(features_train, labels_train)

predictions = my_classifier.predict(features_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, predictions))
