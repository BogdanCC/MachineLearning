from sklearn import tree
from sklearn.naive_bayes import GaussianNB
# Supervised Learning - we give examples(features and correct label) to a program to learn from. Then we can give it data (features) to make predictions for us based on what it learned
    # 1.Collect training data - 2. Train Classifier - 3. Make predictions

# Step 1 : Collecting our data
    # an array of arrays with the features/properties and another array with the correct label for each set of features
    # scikit-learn uses real-valued features so we will need to use ints for our data
    # the first number is the weight(grams) of the fruit and the second is the texture of the fruit
    # [weight, texture] - weight can be any number, texture can be 0("bumpy") or 1 ("smooth")
    # [fruit] - in the labels array we have our fruits   0("apple") or 1 ("orange")

features = [[140,1], [130, 1], [150, 0], [170, 0]]
labels = [0,0,1,1]

# Step 2 : Use our examples(data) to train a classifier
    # The classifier finds patterns in our data by itself(by using the fit function) and creates "rules" to classify our inputs (or predict results)
    # fit() is the training algorithm defined in sklearn

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Step 3 : Now that we have trained our classifier, we can give it inputs to make predictions for us(to label them)
    # we give inputs by calling the predit() function and passing as arguments new features (so an array with 2 bits of data in our case)
    # the classifier will label each set of features based on what it learned
test_data = [[160, 0], [130, 0], [144, 1]]
test_labels = [1, 1, 0]
print("\nCorrect results : \n", *test_labels)
print("Program predicts(Tree clf) : \n", *clf.predict(test_data), "\n\n---------------------\n")
# Creating my own classifier - it should predict if a user liked a video or not based on 5 features we'll give it
# Based on these features, the label will simply be either liked or not liked (represented by 1 and 0 accordingly)
"""
    Features:
        0-100 - precentage of video watched (like 0% or 100%)
        0-1   - if user liked or not (0 no, 1 liked)
        0-1   - if he watched other related videos (0 no 1 yes)
        0-2   - if he skipped through the video and left without rewinding and watching it fully
                (0 he left, never returned; 1 he watched(i.e. didn't skip through while he watched); 2 he skipped through but returned)
        0-1   - if he fullscreened the video (0 no 1 yes)

    Labels:
        1 - liked
        0 - not liked
"""
# Create data - features and their correct labels
features2 = [[30, 0, 1, 2, 0], [28, 0, 0, 2, 0], [68, 0, 1, 1, 0], [92, 1, 1, 1, 1],
             [67, 1, 1, 2, 0], [24, 0, 1, 0, 0], [94, 0, 0, 1, 1], [53, 0, 0, 0, 0]]
labels2 = [0, 0, 0, 1, 1, 0, 1, 0]
# Train the classifier with our dataw
clfG = GaussianNB().fit(features2, labels2)
clfT = tree.DecisionTreeClassifier().fit(features2, labels2)
# Label the features we're passing in
test_data = [[22,0,1,2,0], [82,1,1,1,1], [95,1,0,1,1], [32,0,0,0,0], [50, 1, 0, 0, 0], [50, 0, 0, 0, 0]]
test_label = [0,1,1,0,1,0]
clfG = clfG.predict(test_data)
clfT = clfT.predict(test_data);
# Print Results
print("My correct results : \n", *test_label)
print("(Dec. Tree) Predicted Results are : \n", *clfT)
print("(Naive Bayes) Predicted Results are : \n", *clfG)
