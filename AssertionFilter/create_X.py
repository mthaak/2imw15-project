import csv
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy
import pickle

"""
    Creates a tweets_X.pickle from the tweets.csv to test the AssertionFilter with.
    This file contains the feature matrix for classification of the tweets.
"""

tweets = []
# Read tweet data and labels from file
with open("../Data/tweets_brexit.csv", encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):
        tweets.append(row)

# Convert list to Numpy arrays
tweets = np.array(tweets)

# Determine features (at the moment only word counts, 3 refers to the column number of the tweet message)
X = CountVectorizer().fit_transform(tweets[:, 3])

# Convert to sparse matrix to preserve space
X_sparse = scipy.sparse.csr_matrix(X)

# Pickle to file
with open("../Data/tweets_brexit_X.pickle", "wb") as file:
    pickle.dump(X_sparse, file)

print("File 'tweets_X.pickle' successfully created.")
