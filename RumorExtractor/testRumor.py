from RumorExtractor.rumorExtractor import *
import sklearn
import math
from textblob import TextBlob as tb
import nltk
nltk.download('punkt')
import itertools
import numpy as np
import csv

READ_FILENAME= "../Data/tweets_brexit.csv"
READ_TESTSET= "../Data/test.csv"
WRITE_FILENAME="../Data/tweets_clustered.csv"
# Data of clusters
clusters = []
# Data of output tweets
init_tweets = []

# Import csv file with tweets
re = RumorExtractor()
with open(READ_TESTSET, encoding='utf-8') as csv_file:
    reader1 = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader1):
        init_tweets.append(tb(row[0]))

# Test data
for i in range(10):
    print(init_tweets[i].tags)

# Print status report
print("Filter Tweets")
index_cluster = 0
# Tweets are filtered to only contain verbs and nouns TODO: see further what type of words need to be added / removed
for tweet in init_tweets:
    filtered = ""
    for i in range(len(tweet.words)):
        try:
            if tweet.tags[i][1] == 'VB' or tweet.tags[i][1] == 'VBD' or \
                tweet.tags[i][1] == 'VBG' or tweet.tags[i][1] == 'VBN' or \
                tweet.tags[i][1] == 'VBP' or tweet.tags[i][1] == 'VBZ' or \
                tweet.tags[i][1] == 'NN' or tweet.tags[i][1] == 'NNS' or \
                tweet.tags[i][1] == 'NNP' or tweet.tags[i][1] == 'NNPS':
                    filtered = filtered + tweet.tags[i][0] + " "
        except:
            print("Error")
    # Clusters is represented as a tuple (i, j) where i is the string of merged tweets and j their indexes
    clusters.append([tb(filtered), [index_cluster]])
    index_cluster += 1

# Print status report
print("Tweets filtered")
print(clusters)
# TF-IDF scores of the filtered tweets
t_tfidf = []
# TF-IDF scores of the final clusters
c_tfidf = []
# Keep track of a threshold TODO: look for better value of this threshold
threshold = 0.9
# keep track of the maximum value in the similarity matrix. Init 1.0
max_val = 1.0
# Keep track of the number of times this while loop is entered
clustering = 0
# Keep track of the number of clusters
n_clusters = len(clusters)
# Keep track of the number of tweets
n_tweets = len(init_tweets)

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
        #print("Words in document {}".format(i + 1))
        vector = {word: re.tfidf(word, cluster[0], clusters) for word in cluster[0].words}
        # sorted_words = sorted(vector.items(), key=lambda x: x[1], reverse=True)
        tfidfs.append(vector)
        # Store the TF-IDF scores of the initial filtered tweets and of the TF-IDF scores of the final clusters
        if clustering == 1:
            t_tfidf.append(vector)
        c_tfidf.append(vector)
        #for word, score in vector.items(): # to sort in order of vector, replace with sorted_words.
            #print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

    # Compute the similarity between each pair of clusters and store it in the similarity matrix.
    for i in range(n_clusters):
        for j in range(i+1,n_clusters):
            simMatrix[i][j] = re.computeSimilarity(tfidfs[i], tfidfs[j])
            #print("Similarity between Cluster{} and Cluster{}: {}".format(i, j, simMatrix[i][j]))

    # Convert the similarity matrix into a numpy array
    simMatrix = np.array(simMatrix)
    # Compute the maximum value in the similarity matrix which is inside indexes 0 and 1 of index of argmax
    i1 = np.unravel_index(simMatrix.argmax(), simMatrix.shape)[0] - 1
    i2 = np.unravel_index(simMatrix.argmax(), simMatrix.shape)[1] - 1
    # Merge the clusters with maximum similarity
    clusters = re.mergeClusters(clusters, clusters[i1], clusters[i2])
    # Update values
    n_clusters = len(clusters)
    max_val = simMatrix.max()

# Flatten the list within lists
for i in range(n_clusters):
    indexes = re.flatten(clusters[i][1], [])
    clusters[i] = indexes

print(clusters)
# Print status
print("Tweets Clustered")
print("Find centers of clusters")
# Keep track of the centers of each cluster
centers = []
# Find the tweet in the center that has the most words in common in it, is the rumour
for i in range(n_clusters):
    # Keep track of the maximum similarity
    max_sim = 0
    # Keep track of the index of the tweet that has the maximum similarity
    i_tweet = 0
    for j in range(len(clusters[i])):
        sim = re.computeSimilarity(c_tfidf[i], t_tfidf[clusters[i][j]])
        if sim > max_sim:
            max_sim = sim
            i_tweet = clusters[i][j]
    centers.append(i_tweet)

# Output the set of clusters
# TODO: is output correct for next component?
with open(WRITE_FILENAME, 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter='\t')
    for i in range(n_clusters):
        writer.writerow(str(clusters[i]) + " ," + str(centers[i]))

print("Finished with {} clusters".format(n_clusters))