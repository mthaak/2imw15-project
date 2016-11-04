import csv
import pickle
import scipy.sparse as sp
from TweetEnricher.tweetEnricher import TweetEnricher
import datetime
import numpy as np


READ_FILENAME="../Data/tweets_20161024_111847_assertionlabeled.csv"
WRITE_FILENAME="../Data/Features_20161024"
WRITE_BASIC_FILENAME="../Data/Features_basic_20161024"
WRITE_BINARY_FILENAME = "../Data/Features_binary_20161024"
WRITE_N_GRAM_MATRIX_FILENAME = "../Data/Collection_N_Gram_Matrix_20161024"


tweet_enricher = TweetEnricher()
basic_enriched_tweets = [] #tweets without n gram features
enriched_tweets = [] # tweets with n_gram features and positive,negative opinion counts
binary_enriched_tweets = [] # tweets with all features binary
collection=[] # stores a document of all the tweets


# Read tweet data from file
with open(READ_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):
        #build document for n-grmas count
        if i > 0:
            collection.append(row[1])

#create n-gram count matrix and get list of features(basic and with n grams)
basic_tweet_features,tweet_features,collection_n_gram_count_matrix = tweet_enricher.createNGramCountMatrix(collection)

# collection n-gram count matrix(uni,bi and tri grams)
with open(WRITE_N_GRAM_MATRIX_FILENAME, 'wb') as out_file:
    pickle.dump(collection_n_gram_count_matrix,out_file)

print("Collection's n gram matrix generated -"+  str(datetime.datetime.now().time()))

#get features for each tweet in collection
for tweet in collection:
    # enrich tweets and store in list
    row_enriched_tweets, row_basic_enriched_tweets, row_binary_features = tweet_enricher.enrichTweets(tweet)
    enriched_tweets.append(row_enriched_tweets)
    basic_enriched_tweets.append(row_basic_enriched_tweets)
    binary_enriched_tweets.append(row_binary_features)

np_enriched_tweets = np.array(enriched_tweets)
np_basic_enriched_tweets = np.array(basic_enriched_tweets)
np_binary_enriched_tweets = np.array(binary_enriched_tweets)


print("Feature matrices generated-"+  str(datetime.datetime.now().time()))

#tweet-basic_feature matrix pickeld
with open(WRITE_BASIC_FILENAME, 'wb') as out_file:
    pickle.dump(sp.csr_matrix(np_basic_enriched_tweets),out_file)

print("Basic features pickeled-"+  str(datetime.datetime.now().time()))


#tweet-enriched_feature matrix pickeld
with open(WRITE_FILENAME, 'wb') as out_file:
    pickle.dump(sp.csr_matrix(np_enriched_tweets),out_file)

print("N grams and pos/neg counts' features pickeled-"+  str(datetime.datetime.now().time()))

#tweet-enriched_feature matrix pickeld
with open(WRITE_BINARY_FILENAME, 'wb') as out_file:
    pickle.dump(sp.csr_matrix(np_binary_enriched_tweets),out_file)

print("All binary features pickeled-"+  str(datetime.datetime.now().time()))
