import csv
import numpy as np
import pickle
import scipy
from sklearn import *
from assertionfilter import *

TWEETS_FILEPATH = "../Data/tweets_20161024_111847_assertionlabeled.csv"
FEATURES_FILEPATH = "../Data/04_11/Features_binary.pickle"
FILTERED_TWEETS_FILEPATH = "../Data/tweets_20161024_111847_assertionfiltered.csv"
FILTERED_FEATURES_FILEPATH = "../Data/Features_binary_assertionfiltered.pickle"
tweets = []  # data of tweets
labels = []  # labels of tweets
indices_labeled = []  # indices of labeled tweets
indices_unlabeled = []  # indices of unlabeled tweets

# Read tweet data from file
with open(TWEETS_FILEPATH, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    header = next(reader)  # skip header
    for i, row in enumerate(reader):
        tweets.append(row)
        # 0 = not assertion, 1 = assertion, -1 = <unlabeled>
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

# Turn lists into the more convenient Numpy array
tweets = np.array(tweets)
labels = np.array(labels)

print("There are", len(indices_unlabeled), "unlabeled tweets and", len(indices_labeled), "labeled tweets.")

# Get feature matrix from file
with open(FEATURES_FILEPATH, "rb") as file:
    X = pickle.load(file)
features = X.toarray()

# Initialize assertion filter with desired classifier
assertion_filter = AssertionFilter(
    sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_features=None, n_jobs=2))

# Train classifier
assertion_filter.train(features[indices_labeled], labels[indices_labeled])

# Classify
predicted_labels = assertion_filter.classify(features[indices_unlabeled])

# Set predicted labels
for i, index in enumerate(indices_unlabeled):
    labels[index] = predicted_labels[i]

# Filter tweets
filtered_tweets, filtered_features = assertion_filter.filter(tweets, features, labels)

# Remove last column (is_assertion)
del header[-1]
filtered_tweets = np.delete(filtered_tweets, filtered_tweets.shape[1] - 1, 1)

print("Of the total", len(tweets), "tweets", len(filtered_tweets), "are left after filtering.")

# Write filtered tweets to file
with open(FILTERED_TWEETS_FILEPATH, 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter='\t')
    writer.writerow(header)
    for tweet in filtered_tweets:
        writer.writerow(tweet)

# Write filtered feature matrix to file
with open(FILTERED_FEATURES_FILEPATH, 'wb') as pickle_file:
    pickle.dump(scipy.sparse.csr_matrix(filtered_features), pickle_file)
