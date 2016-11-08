from RumorExtractor.rumorExtractor import *
import sklearn
import math
from textblob import TextBlob as tb
import nltk
nltk.download('punkt')
import numpy as np
import csv

READ_TESTSET= "../Data/test.csv"
READ_FILENAME= "../Data/tweets_20161024_111847_assertionfiltered.csv"
WRITE_FILENAME="../Data/tweets_20161024_111847_clustered.csv"

# Data of clusters
clusters = []

# Import csv file with tweets
re = RumorExtractor()
with open(READ_FILENAME, encoding='utf-8') as csv_file:
    reader1 = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader1):
        clusters.append([tb(row[1]), row[0]])

# Print status report
print("Filter Tweets")
# The tags of the words that are not filtered out in the tweets
tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP', 'NNPS', 'JJ']
# The words that do need to be filtered out
filter_out = ['@', 'RT', '[', ']', 'http', 'https', 'urls']

# Tweets are filtered to only contain verbs and nouns
for tweet in clusters:
    filtered = ""
    for i in range(len(tweet[0].words)):
        for j in range(len(tags)):
            try:
                if tweet[0].tags[i][1] == tags[j] and not (tweet[0].tags[i][0] in filter_out):
                    filtered = filtered + tweet[0].tags[i][0] + " "
            except:
                print("Error")
    # Clusters is represented as a tuple (i, j) where i is the string of merged tweets and j their Tweet ID
    clusters.append([tb(filtered), [tweet[1]]])
    clusters.remove(tweet)

# Print status report
print("Tweets filtered")
print(clusters)
# TF-IDF scores of the filtered tweets
t_tfidf = []
# TF-IDF scores of the final clusters
c_tfidf = []
# Keep track of a threshold
threshold = 0.045
# keep track of the maximum value in the similarity matrix. Init 1.0
max_val = 1.0
# Keep track of the number of times this while loop is entered
clustering = 0
# Keep track of the number of clusters
n_clusters = len(clusters)
# Print status
print("Cluster tweets")

# Keep clustering until threshold is reached or when there is only one cluster left.
while max_val > threshold and n_clusters > 1:
    clustering += 1
    # The TF-IDF scores of all tweets
    tfidfs = []
    # Similarity matrix with size n x n clusters all set to 0.
    simMatrix = [[0.0 for x in range(n_clusters)] for y in range(n_clusters)]

    # Compute the TF-IDF vector for each of the tweets
    for i, cluster in enumerate(clusters):
        vector = {word: re.tfidf(word, cluster[0], clusters) for word in cluster[0].words}
        tfidfs.append(vector)
        # Store the TF-IDF scores of the initial filtered tweets and of the TF-IDF scores of the final clusters
        if clustering == 1:
            t_tfidf.append(vector)
        c_tfidf.append(vector)
        print(i)

    # Compute the similarity between each pair of clusters and store it in the similarity matrix.
    for i in range(n_clusters):
        for j in range(i+1,n_clusters):
            simMatrix[i][j] = re.computeSimilarity(tfidfs[i], tfidfs[j])

    # Convert the similarity matrix into a numpy array
    simMatrix = np.array(simMatrix)
    # Compute the maximum value in the similarity matrix which is inside indexes 0 and 1 of index of argmax
    i1 = np.unravel_index(simMatrix.argmax(), simMatrix.shape)[0]
    i2 = np.unravel_index(simMatrix.argmax(), simMatrix.shape)[1]
    # Merge the clusters with maximum similarity
    clusters = re.mergeClusters(clusters, clusters[i1], clusters[i2])
    # Update values
    n_clusters = len(clusters)
    max_val = simMatrix.max()
    print(max_val)
    print("There are still {} clusters".format(n_clusters))

print(clusters)

# Flatten the list within lists
for i in range(n_clusters):
    indexes = re.flatten(clusters[i][1], [])
    clusters[i] = indexes

# Print status
print("Tweets Clustered")
print(clusters)
print("Find centers of clusters")
# Keep track of the centers of each cluster
centers = []
# Find the tweet in the center that has the most words in common in it, is the rumour
for i in range(n_clusters):
    # Keep track of the maximum similarity
    max_sim = 0
    # Keep track of the index of the tweet that has the maximum similarity
    i_tweet = -1
    for j in range(len(clusters[i])):
        sim = re.computeSimilarity(c_tfidf[i], t_tfidf[clusters[i][j]])
        if sim > max_sim:
            max_sim = sim
            i_tweet = clusters[i][j]
    if i_tweet == -1:
        i_tweet = clusters[i][0]
    centers.append(i_tweet)

# Output the set of clusters
with open(WRITE_FILENAME, 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter='\t')
    for i in range(n_clusters):
        writer.writerow(str(clusters[i]) + " ," + str([centers[i]]))

print("Finished with {} clusters".format(n_clusters))