from assertionfilter import *
import numpy as np
import csv
import scipy
from sklearn import *
import pickle
import datetime

TWEETS_FILENAME = "../Data/tweets_20161024_111847_assertionlabeled.csv"
X_FILENAME = "../Data/Features_basic"
RESULTS_FILENAME = "./evaluate_classifiers_results.csv"

# Get labels from tweets.csv file
labels = []
indices_labeled = []  # indices of labeled tweets
indices_unlabeled = []  # indices of unlabeled tweets
with open(TWEETS_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    next(reader)  # skip header
    for i, row in enumerate(reader):
        # 0 = NR, 1 = R, 2 = U, -1 = <unlabeled>
        if row[-1] == '0':
            labels.append(0)
        elif row[-1] == '1':
            labels.append(1)
        else:
            labels.append(-1)

        if row[-1] in ['0', '1']:
            indices_labeled.append(i)
        else:
            indices_unlabeled.append(i)

# Convert list to the more convenient Numpy array
y = np.array(labels)

# Get feature matrix from file
with open(X_FILENAME, "rb") as file:
    X = pickle.load(file)
X = X.toarray()

### CLASSIFIERS ###
classifiers_with_parameters = []

# dummy
# DummyClassifier
classifier = sklearn.dummy.DummyClassifier(strategy='most_frequent')
parameters = [
    {}
]
# classifiers_with_parameters.append((classifier, parameters))

# ensemble

# BaggingClassifier
classifier = sklearn.ensemble.BaggingClassifier()
parameters = [
    {}
]
# classifiers_with_parameters.append((classifier, parameters))

# RandomForestClassifier
classifier = sklearn.ensemble.RandomForestClassifier()
parameters = [
    {}
]
# classifiers_with_parameters.append((classifier, parameters))

# discriminant_analysis
# LinearDiscrimantAnalysis
classifier = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
parameters = [
    {}
]
# classifiers_with_parameters.append((classifier, parameters))

# linear_model
# LogisticRegression
classifier = sklearn.linear_model.LogisticRegression()
parameters = [
    {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag"]
    }
]
# classifiers_with_parameters.append((classifier, parameters))

# naive_bayes
# GuassianNB
classifier = sklearn.naive_bayes.GaussianNB()
parameters = [
    {}
]
# classifiers_with_parameters.append((classifier, parameters))

# MultinomialNB
classifier = sklearn.naive_bayes.MultinomialNB()
parameters = [
    {}
]
# classifiers_with_parameters.append((classifier, parameters))

# BernoulliNB
classifier = sklearn.naive_bayes.BernoulliNB()
parameters = [
    {"binarize": [False]}
]
# classifiers_with_parameters.append((classifier, parameters))

# neighbors
# KNeighborsClassifier
classifier = sklearn.neighbors.KNeighborsClassifier()
parameters = [
    {
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "brute"],
        "metric": ["minkowski", "hamming", "canberra", "braycurtis"]
    },
    {
        "weights": ["uniform", "distance"],
        "algorithm": ["kd_tree"],
    }
]
classifiers_with_parameters.append((classifier, parameters))

# RadiusNeighborsClassifier
classifier = sklearn.neighbors.RadiusNeighborsClassifier()
parameters = [
    {
        # "weight": ["uniform", "distance"],
        # "algorithm": ["ball_tree", "kd_tree", "brute"],
        # "metric": ["minkowski", "hamming", "canberra", "braycurtis"]
    }
]
# classifiers_with_parameters.append((classifier, parameters))

# svm
# SVC
classifier = sklearn.svm.SVC()
parameters = [
    {
        # "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    }
]
# classifiers_with_parameters.append((classifier, parameters))

# LinearSVC
classifier = sklearn.svm.LinearSVC()
parameters = [
    {}
]
# classifiers_with_parameters.append((classifier, parameters))

# tree
# DecisionTreeClassifier
classifier = sklearn.tree.DecisionTreeClassifier()
parameters = [
    {}
]


# classifiers_with_parameters.append((classifier, parameters))


def write_results_to_file(clf, scorings, grid_scores):
    with open(RESULTS_FILENAME, 'a', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')

        writer.writerow(["time: " + str(datetime.datetime.now())])

        parameter_keys = list(clf.grid_scores_[0].parameters.keys())
        header = scorings + ["classifier"] + parameter_keys
        writer.writerow(header)

        classifier_name = clf.estimator.__class__.__name__
        parameter_values = map(lambda x: list(x.parameters.values()), clf.grid_scores_)
        for scores, parameters in zip(grid_scores, parameter_values):
            writer.writerow(scores + [classifier_name] + parameters)


grid_scores = []


def scorer(truth, pred):
    f1_score = sklearn.metrics.f1_score(truth, pred, average='macro')

    def to_list(a):
        return [str(a[0][0]), str(a[0][1]), str(a[1][0]), str(a[1][1])]

    grid_scores.append(
        [f1_score,
         sklearn.metrics.precision_score(truth, pred),
         sklearn.metrics.recall_score(truth, pred),
         sklearn.metrics.accuracy_score(truth, pred)]
        + to_list(sklearn.metrics.confusion_matrix(truth, pred))
    )

    return f1_score  # optimize for f1 score


scorings = ['f1', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'fn', 'tn']
my_scorer = sklearn.metrics.make_scorer(scorer, greater_is_better=True)

# Evaluate classifiers with different parameters
for (classifier, parameters) in classifiers_with_parameters:
    try:
        clf = sklearn.grid_search.GridSearchCV(classifier, parameters, cv=10, scoring=my_scorer)
        clf.fit(X[indices_labeled], y[indices_labeled])
        write_results_to_file(clf, scorings, grid_scores)
    except Exception as err:
        pass
