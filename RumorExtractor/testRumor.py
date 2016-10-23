from rumorExtractor import *
import sklearn
import math
from textblob import TextBlob as tb
import nltk
nltk.download('punkt')
import numpy as np
import csv

READ_FILENAME= "../Data/tweets_filtered.csv"
WRITE_FILENAME="../Data/tweets_clustered.csv"
# Data of clusters
clusters = []
# Data of output tweets
init_tweets = []

# Import csv file with tweets
re = RumorExtractor()
with open(READ_FILENAME, encoding='utf-8') as csv_file:
    reader1 = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader1):
        init_tweets.append(tb(row[3]))

# TODO: remove unneeded comments like print, they are used for testing
#tweet1 = tb("#zikavirus	TheHealthZoneNg	3081513976	can the effects of #Zikavirus be more neurological ? https://t.co/aDoC0Yh")
#tweet2 = tb("#zikavirus	ChrisSaldana	74753973	Fish helping take on the #ZikaVirus. The #Texas city trying the fish out 4:45am on")
#tweet3 = tb("#zikavirus	srritchotte	709736512041017000	Nice hope they did not bring the #ZikaVirus with that")
#init_tweets = [tweet1, tweet2, tweet3]

# Tweets are filtered to only contain verbs and nouns TODO: discuss further what type of words need to be added / removed
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
    clusters.append(tb(filtered))
print("Tweets filtered")

# TF-IDF scores of the filtered tweets
t_tfidf = []
# TF-IDF scores of the final clusters
c_tfidf = []
# Keep track of a threshold TODO: look for true_k
threshold = 0.001
# keep track of the maximum value in the similarity matrix. Init 1.0
max_val = 1.0
# Keep track of the number of times this while loop is entered
clustering = 0
# Keep track of the number of clusters
n_clusters = len(clusters)
# Keep track of the number of tweets
n_tweets = len(init_tweets)

# Keep clustering until threshold is reached or when there is only one cluster left.
while max_val > threshold and n_clusters > 1:
    clustering = clustering + 1
    # The TF-IDF scores of all tweets
    tfidfs = []
    # Similarity matrix with size n x n clusters all set to 0.
    simMatrix = [[0.0 for x in range(n_clusters)] for y in range(n_clusters)]

    # Compute the TF-IDF vector for each of the tweets
    for i, cluster in enumerate(clusters):
        #print("Words in document {}".format(i + 1))
        vector = {word: re.tfidf(word, cluster, clusters) for word in cluster.words}
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
        for j in range(n_clusters):
            if i < j :
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

# Keep track of the centers of each cluster
centers = []
# Find the tweet in the center that has the most words in common in it, is the rumour
for i in range(n_clusters):
    # Keep track of the maximum similarity
    max_sim = 0
    # Keep track of the index of the tweet that has the maximum similarity
    i_tweet = 0
    for j in range(n_tweets):
        sim = re.computeSimilarity(c_tfidf[i], t_tfidf[j])
        if sim > max_sim:
            max_sim = sim
            i_tweet = j
    centers.append(i_tweet)

# Output the set of clusters
#for i in range(n_clusters):
#    print([clusters[i], centers[i]])
# TODO: output array of indexes of tweets that belong to each cluster
with open(WRITE_FILENAME, 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter='\t')
    for i in range(n_clusters):
        writer.writerow([clusters[i].string + ",", centers[i]])

print("Finish")