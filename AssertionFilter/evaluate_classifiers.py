from assertionfilter import *
import numpy as np
import csv
import scipy
from sklearn import *
import pickle

TWEETS_FILENAME = "../Data/tweets.csv"
X_FILENAME = "../Data/tweets_X.pickle"
RESULTS_FILENAME = "./evaluate_classifiers_results.csv"

# Get labels from tweets.csv file
labels = []
indices_labeled = []  # indices of labeled tweets
indices_unlabeled = []  # indices of unlabeled tweets
with open(TWEETS_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):
        # 0 = NR, 1 = R, 2 = U, -1 = <unlabeled>
        if row[10] == 'NR':
            labels.append(0)
        elif row[10] == 'R':
            labels.append(1)
        elif row[10] == 'U':
            labels.append(2)
        else:
            labels.append(-1)

        if row[10] in ['NR', 'R', 'U']:
            indices_labeled.append(i)
        else:
            indices_unlabeled.append(i)

# Convert list to the more convenient Numpy array
y = np.array(labels)[indices_labeled]

# Get feature matrix from file
with open(X_FILENAME, "rb") as file:
    X = pickle.load(file)[indices_labeled]

### CLASSIFIERS ###
# AdaBoostClassifier
classifier = sklearn.ensemble.AdaBoostClassifier()
parameters = [
    {"n_estimators": [25, 50, 100],
     "learning_rate": [i / 10 for i in range(1, 25, 5)],
     "algorithm": ["SAMME.R"]}
]

# DummyClassifier
# classifier = sklearn.dummy.DummyClassifier(strategy='most_frequent')
# parameters = []

# MultinomialNB
# classifier = sklearn.naive_bayes.MultinomialNB()

# LinearSVC
# classifier = sklearn.svm.LinearSVC()

# RandomForestClassifier
# classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=10)

# LogisticRegression
# classifier = sklearn.linear_model.LogisticRegression()

# Evaluate classifier with different parameters
clf = sklearn.grid_search.GridSearchCV(classifier, parameters, cv=10, scoring='f1_macro')
clf.fit(X, y)

# Write results to file
with open(RESULTS_FILENAME, 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    header = ["score", "classifier"] + list(clf.best_params_.keys())
    classifier_name = clf.estimator.__class__.__name__
    writer.writerow(header)
    for result in clf.grid_scores_:
        writer.writerow([result.mean_validation_score, classifier_name] + list(result.parameters.values()))
