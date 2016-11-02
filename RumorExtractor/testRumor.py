from RumorExtractor.rumorExtractor import *
from textblob import TextBlob as tb
import nltk
import numpy as np

# Make sure punkt corpus is up-to-date!
nltk.download('punkt')

# Import csv file TODO: still needs to import actual csv instead of these tests
tweets = []  # data of tweets
tweets.append(tb("RT @ClassicPict: Mosquitoes kill more annually than Sharks. #ZikaV"))
tweets.append(
    tb("@kanikahanda Pakaging matters however good or bad the product is & star performer this time is #ZikaVirus"))

# Initialize rumor extractor
re = RumorExtractor()
threshold = 0.001  # Keep track of a threshold
max_val = 1.0  # keep track of the maximum value in the similarity matrix. Init 1.0
n_tweets = len(tweets)

# keep clustering until threshold is reached or when there is only one cluster left.
while max_val > threshold and n_tweets > 1:
    tfidfs = []  # The tfidf scores of all tweets
    simMatrix = [[0 for x in range(n_tweets)] for y in
                 range(n_tweets)]  # Similarity matrix with size n x n tweets all set to 0.

    # Compute the TF-IDF vector for each of the tweets
    for i, tweet in enumerate(tweets):
        print("Words in document {}".format(i + 1))
        vector = {word: re.tfidf(word, tweet, tweets) for word in tweet.words}
        # sorted_words = sorted(vector.items(), key=lambda x: x[1], reverse=True)
        tfidfs.append(vector)
        for word, score in vector.items():  # to sort in order of vector, replace with sorted_words.
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

    # Compute the similarity between each pair of clusters and store it in the similarity matrix.
    for i in range(n_tweets):
        for j in range(n_tweets):
            if i == j:
                simMatrix[i][j] = 0.0
            else:
                simMatrix[i][j] = re.computeSimilarity(tfidfs[i], tfidfs[j])
            print("Similarity between Tweet{} and Tweet{}: {}".format(i, j, simMatrix[i][j]))

    # Convert the similarity matrix into a numpy array
    simMatrix = np.array(simMatrix)
    # Compute the maximum value in the similarity matrix which is inside indexes 0 and 1 of index of argmax
    tweets = re.mergeClusters(tweets, tweets[simMatrix.argmax(axis=0)[0]], tweets[simMatrix.argmax(axis=0)[1]])
    n_tweets = len(tweets)
    max_val = simMatrix.max()

# Output the set of clusters TODO: still need to do the actual outputting of clusters
print(tweets)
