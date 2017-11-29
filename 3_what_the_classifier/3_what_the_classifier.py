# In this program we will use 3 different classifiers and check their accuracy
# The purpose of today is to think of the classifiers as a function

# Import and load toy iris data set
from sklearn import datasets as ds
iris_data = ds.load_iris()

# Now we will name the features and labels with X and y in order to think of this as a function
# f(x) = y ; where x = features and y = labels
X = iris_data.data # features
y = iris_data.target # labels

# Now we can split the data from the iris dataset, so we have our training data and our testing data
# A little handy utility from sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5) # this will split the data in 2 (50% test data, 50% train data)

# Let's use 3 classifiers and print out their accuracy
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Create classifiers
clfT = tree.DecisionTreeClassifier()
clfG = GaussianNB()
clfN = KNeighborsClassifier()

# Train classifiers with train data
clfT.fit(X_train, y_train)
clfG.fit(X_train, y_train)
clfN.fit(X_train, y_train)

# Make predictions from test data
predictionT = clfT.predict(X_test)
predictionG = clfG.predict(X_test)
predictionN = clfN.predict(X_test)
print("Correct answers :\n", y_test)
print("Tree predicts :\n", predictionT)
print("Bayes predicts :\n", predictionG)
print("KNeighbors predicts :\n", predictionN)

# Check accuracy
from sklearn.metrics import accuracy_score
print("Tree accuracy : ", accuracy_score(y_test, predictionT))
print("Bayes accuracy : ", accuracy_score(y_test, predictionG))
print("KNeighbors accuracy : ", accuracy_score(y_test, predictionN))
