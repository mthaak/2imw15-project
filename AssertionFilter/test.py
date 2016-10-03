import csv
import numpy as np
from sklearn import *
from assertionfilter import *

READ_FILENAME = "../Data/tweets.csv"
WRITE_FILENAME = "../Data/filtered_tweets.csv"
tweets = []  # data of tweets
labels = []  # labels of tweets
indices_labeled = []  # indices of labeled tweets
indices_unlabeled = []  # indices of unlabeled tweets

# Read tweet data from file
with open(READ_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):
        tweets.append(row)
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

# Turn lists into the more convenient Numpy array
tweets = np.array(tweets)
labels = np.array(labels)

print("There are", len(indices_unlabeled), "unlabeled tweets and", len(indices_labeled), "labeled tweets.")

# Determine features (at the moment only word counts, 3 refers to the column nr)
features = sklearn.feature_extraction.text.CountVectorizer().fit_transform(tweets[:, 3])

# Initialize assertion filter with desired classifier
assertion_filter = AssertionFilter(sklearn.svm.LinearSVC())

# Train classifier
assertion_filter.train(features[indices_labeled], labels[indices_labeled])

# Classify
predicted_labels = assertion_filter.classify(features[indices_unlabeled])

# Set predicted labels
for i, index in enumerate(indices_unlabeled):
    labels[index] = predicted_labels[i]

# Filter tweets
filtered_tweets, filtered_features, filtered_labels = assertion_filter.filter(tweets, features, labels)

print("Of the total", len(tweets), "tweets", len(filtered_tweets), "are left after filtering.")

# Write filtered tweets to file
with open(WRITE_FILENAME, 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter='\t')
    for tweet in filtered_tweets:
        writer.writerow(tweet)

# FOR TESTING DIFFERENT CLASSIFIERS
# classifiers = [
#     sklearn.dummy.DummyClassifier(strategy='most_frequent'),
#     sklearn.naive_bayes.MultinomialNB(),
#     sklearn.svm.LinearSVC(),
#     sklearn.linear_model.LogisticRegression(),
#     sklearn.ensemble.RandomForestClassifier(n_estimators=10),
#     sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
# ]
#
# for classifier in classifiers:
#     assertion_filter = AssertionFilter(classifier)
#     score = assertion_filter.evaluate(count_matrix, labels)
#     print(classifier, '\033[33m', score, '\033[0m')
