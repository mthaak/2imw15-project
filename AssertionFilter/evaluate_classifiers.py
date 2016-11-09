import numpy as np
import csv
from sklearn import dummy, linear_model, discriminant_analysis, svm, naive_bayes, neighbors, tree, ensemble, metrics, \
    grid_search, pipeline, decomposition, preprocessing
import pickle
import datetime

TWEETS_FILENAME = "../Data/tweets_20161024_111847_assertionlabeled.csv"
X_FILENAME = "../Data/04_11/Features.pickle"
RESULTS_FILENAME = "./04_11_results/evaluate_classifiers_results_0411.csv"

# Get training labels from tweets file
labels = []
indices_labeled = []  # indices of labeled tweets
with open(TWEETS_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    next(reader)  # skip header
    for i, row in enumerate(reader):
        # 0 = not assertion, 1 = assertion, -1 = <unlabeled>
        if row[-1] == '0':
            labels.append(0)
        elif row[-1] == '1':
            labels.append(1)
        else:
            labels.append(-1)

        if row[-1] in ['0', '1']:
            indices_labeled.append(i)

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
classifier = dummy.DummyClassifier(strategy='constant', constant=1)
parameters = [
    {
        # no useful parameters
    }
]
classifiers_with_parameters.append((classifier, parameters))

# linear_model
# LogisticRegression
classifier = linear_model.LogisticRegression()
parameters = [
    {
        # no useful parameters
    }
]
classifiers_with_parameters.append((classifier, parameters))

# svm
# LinearSVC
classifier = svm.LinearSVC()
parameters = [
    {
        # no useful parameters
    }
]
classifiers_with_parameters.append((classifier, parameters))

# naive_bayes
# MultinomialNB
classifier = naive_bayes.MultinomialNB()
parameters = [
    {
        # no useful parameters
    }
]
classifiers_with_parameters.append((classifier, parameters))

# BernoulliNB
classifier = naive_bayes.BernoulliNB()
parameters = [
    {
        # no useful parameters
    }
]
classifiers_with_parameters.append((classifier, parameters))

# discriminant_analysis
# LinearDiscrimantAnalysis + BernoulliNB
classifier = pipeline.make_pipeline(discriminant_analysis.LinearDiscriminantAnalysis(), svm.LinearSVC())
parameters = [
    {
        # no useful parameters
    }
]
classifiers_with_parameters.append((classifier, parameters))

# neighbors
# KNeighborsClassifier
classifier = neighbors.KNeighborsClassifier()
parameters = [
    {
        "weights": ["distance"],
        "metric": ["hamming", "canberra", "braycurtis", "minkowski"]
    }
]
classifiers_with_parameters.append((classifier, parameters))

# tree
# DecisionTreeClassifier
classifier = tree.DecisionTreeClassifier()
parameters = [
    {
        "max_features": [None],
    }
]
classifiers_with_parameters.append((classifier, parameters))

# ensemble
# RandomForestClassifier
classifier = ensemble.RandomForestClassifier()
parameters = [
    {
        "n_estimators": [100],
        "max_features": [None],
    }
]
classifiers_with_parameters.append((classifier, parameters))

### Grid Search ###
n_folds = 10
n_labels = len(indices_labeled)
scorings = ['f1', 'accuracy', 'tp', 'fp', 'fn', 'tn']  # names of scorings
true_glob, pred_glob = [], []


def scorer(true, pred):
    true_glob += true.tolist()
    pred_glob += pred.tolist()
    global true_glob, pred_glob

    return metrics.f1_score(true, pred, average='binary', pos_label=1)  # optimize for f1 score


my_scorer = metrics.make_scorer(scorer, greater_is_better=True)


def avg_over_params(true, pred, n_labels):
    assert len(true) == len(pred)
    avg_param_scores = []

    def to_list(matrix):
        return [matrix[1][1], matrix[0][1], matrix[1][0], matrix[0][0]]

    for i in range(0, len(true), n_labels):
        avg_param_scores.append(
            [metrics.f1_score(true[i: i + n_labels], pred[i: i + n_labels], average='binary'),
             metrics.accuracy_score(true[i: i + n_labels], pred[i: i + n_labels]),
             ] + to_list(metrics.confusion_matrix(true[i: i + n_labels], pred[i: i + n_labels]))
        )

    return avg_param_scores


def write_results_to_file(clf, scorings, grid_scores):
    with open(RESULTS_FILENAME, 'a', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')

        # writer.writerow(["time: " + str(datetime.datetime.now())])

        parameter_keys = list(clf.grid_scores_[0].parameters.keys())
        header = scorings + ["classifier"] + parameter_keys
        writer.writerow(header)

        classifier_name = clf.estimator.__class__.__name__
        parameter_values = map(lambda x: list(x.parameters.values()), clf.grid_scores_)
        for scores, parameters in zip(grid_scores, parameter_values):
            writer.writerow(scores + [classifier_name] + parameters)


# Evaluate classifiers with different parameters
for (classifier, parameters) in classifiers_with_parameters:
    try:
        true_glob, pred_glob = [], []
        clf = grid_search.GridSearchCV(classifier, parameters, cv=n_folds, scoring=my_scorer)
        clf.fit(X[indices_labeled], y[indices_labeled])
        write_results_to_file(clf, scorings, avg_over_params(true_glob, pred_glob, n_labels))
    except Exception as err:
        pass
