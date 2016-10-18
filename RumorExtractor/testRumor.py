from rumorExtractor import *
import sklearn
import math
from textblob import TextBlob as tb
import nltk
nltk.download('punkt')
import numpy as np
import csv

READ_TWEETS= "../Data/tweets_filtered.csv"
READ_RUMOURS= "../Data/rumourslist.csv"
WRITE_FILENAME="../Data/features"
# Data of tweets
tweets = []
# Data of rumours
rumours = []

# Import csv file with tweets
re = RumorExtractor()
with open(READ_TWEETS, encoding='utf-8') as csv_file:
    reader1 = csv.reader(csv_file, delimiter='\t')
    for i1, row2 in enumerate(reader1):
        x = 1
        #tweets.append(tb(row[3]))

tweet1 = tb("#zikavirus	TheHealthZoneNg	3081513976	can the effects of #Zikavirus be more neurological ? https://t.co/aDoC0Yh")
tweet2 = tb("#zikavirus	ChrisSaldana	74753973	Fish helping take on the #ZikaVirus. The #Texas city trying the fish out 4:45am on")
tweet3 = tb("#zikavirus	srritchotte	709736512041017000	Nice hope they did not bring the #ZikaVirus with that")
tweets = [tweet1, tweet2, tweet3]
# Data of output tweets
init_tweets = [tweet1, tweet2, tweet3]
rumour1 = tb("Fish")
rumour2 = tb("Nice")
rumours = [rumour1, rumour2]

# Keep track of a threshold
threshold = 0.001
# keep track of the maximum value in the similarity matrix. Init 1.0
max_val = 1.0
# Keep track of the number of tweets
n_tweets = len(tweets)
# Keep track of the number of rumours
n_rumours = len(rumours)
#
n_init_tweets = len(init_tweets)
# Data of the output tweets, init to -1
out_tweets = []

# keep clustering until threshold is reached or when there is only one cluster left.
while max_val > threshold and n_tweets > 1:
    # The tfidf scores of all tweets
    tfidfs = []
    # Similarity matrix with size n x n tweets all set to 0.
    simMatrix = [[0 for x in range(n_tweets)] for y in range(n_tweets)]

    # Compute the TF-IDF vector for each of the tweets
    for i, tweet in enumerate(tweets):
        print("Words in document {}".format(i + 1))
        vector = {word: re.tfidf(word, tweet, tweets) for word in tweet.words}
        # sorted_words = sorted(vector.items(), key=lambda x: x[1], reverse=True)
        tfidfs.append(vector)
        for word, score in vector.items(): # to sort in order of vector, replace with sorted_words.
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

    # Compute the similarity between each pair of clusters and store it in the similarity matrix.
    for i in range(n_tweets):
        for j in range(n_tweets):
            if i == j:
                simMatrix[i][j] = 0.0
            else :
                simMatrix[i][j] = re.computeSimilarity(tfidfs[i], tfidfs[j])
            print("Similarity between Tweet{} and Tweet{}: {}".format(i, j, simMatrix[i][j]))

    # Convert the similarity matrix into a numpy array
    simMatrix = np.array(simMatrix)
    # Compute the maximum value in the similarity matrix which is inside indexes 0 and 1 of index of argmax
    # TODO: merging and outputting must be differently
    i1 = np.unravel_index(simMatrix.argmax(), simMatrix.shape)[0] - 1
    i2 = np.unravel_index(simMatrix.argmax(), simMatrix.shape)[1] - 1
    tweets = re.mergeClusters(tweets, tweets[i1], tweets[i2])
    n_tweets = len(tweets)
    max_val = simMatrix.max()
    print(tweets)

print(init_tweets)

# First find which rumour is in which cluster TODO: volgorde for loop om out_tweets niet onnodig groot te maken
for i in range(n_rumours):
    for j in range(n_tweets):
        print(rumours[i].string)
        if rumours[i].string in tweets[j].string:
            for k in range(n_init_tweets):
                out_tweets.append([])
                if init_tweets[k].string in tweets[j].string:
                    out_tweets[k].append(i)

# Then find which tweets are in which cluster with this rumour
print(out_tweets)
print("Finish")

# Output the set of clusters TODO: still need to do the actual outputting of clusters
#print(tweets)