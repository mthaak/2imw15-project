import csv
import pickle
import scipy.sparse as sp
from TweetEnricher.tweetEnricher import TweetEnricher
import datetime
import numpy as np


READ_FILENAME="../Data/tweets_ 20161024_111847_assertionlabeled.csv"
WRITE_FILENAME="../Data/Features.pickle"
WRITE_BASIC_FILENAME="../Data/Features_basic.pickle"
WRITE_BINARY_FILENAME = "../Data/Features_binary.pickle"
WRITE_N_GRAM_MATRIX_FILENAME = "../Data/Collection_N_Gram_Matrix.pickle"
WRITE_SPEECH_ACT_TAG_FILENAME = "../Data/Collection_Speech_Act_Tagged.pickle"


tweet_enricher = TweetEnricher()
basic_enriched_tweets = [] #tweets without n gram features
enriched_tweets = [] # tweets with n_gram features and positive,negative opinion counts
binary_enriched_tweets = [] # tweets with all features binary
collection=[] # stores a list of all the tweets
collection_tweets = {} # stores all tweet ids
collection_urls = {} # stores the tweet urls

# Read tweet data from file
with open(READ_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):
        #build document for n-grmas count
        if i > 0:
            collection.append(row[1])
            collection_tweets[i] = row[1]
            collection_urls[i] = row[11]


# speech_act_tags = tweet_enricher.speechActTagCollection(collection_tweets)
#
# # speech act tagged collection
# with open(WRITE_SPEECH_ACT_TAG_FILENAME, 'wb') as out_file:
#     pickle.dump(speech_act_tags, out_file)
#
print("Collection - speech act tagged -"+ str(datetime.datetime.now().time()))

#When reading pre-tagged collection
SA_tagged_collection = pickle.load(open(WRITE_SPEECH_ACT_TAG_FILENAME, "rb"))

#create n-gram count matrix and get list of features(basic and with n grams)
basic_tweet_features,tweet_features,collection_n_gram_count_matrix = tweet_enricher.createNGramCountMatrix(collection,SA_tagged_collection)

# #collection n-gram count matrix(uni,bi and tri grams)
# with open(WRITE_N_GRAM_MATRIX_FILENAME, 'wb') as out_file:
#     pickle.dump(collection_n_gram_count_matrix,out_file)
#
# print("Collection's n gram matrix generated -"+  str(datetime.datetime.now().time()))
#
# collection_n_gram_count_matrix_from_pickle = pickle.load(open(WRITE_N_GRAM_MATRIX_FILENAME, "rb"))


#get features for each tweet in collection
for tweet in collection_tweets:
    # enrich tweets and store in list
    row_enriched_tweets, row_basic_enriched_tweets, row_binary_features = tweet_enricher.enrichTweets(collection_tweets.get(tweet),collection_urls.get(tweet))
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
