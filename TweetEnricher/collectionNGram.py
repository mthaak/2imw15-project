import csv
import pickle
from TweetEnricher.tweetEnricher import TweetEnricher
import datetime

"""
    Tags tweets with their speech tags (if any) and stores in Data/Collection_Speech_Act_Tagged as a pickle.
    A collection wide n-gram (uni, bi and tri) matrix is created and stored in Data/Collection_N_Gram_Matrix as a pickle.
"""

READ_FILENAME="../DataCollection/results/search_20161024_111847_tweets.csv"
WRITE_FILENAME="../Data/test/N_Gram_Features"
WRITE_BASIC_FILENAME="../Data/Features_basic"
WRITE_BINARY_FILENAME = "../Data/Features_binary"
WRITE_N_GRAM_MATRIX_FILENAME = "../Data/Collection_N_Gram_Matrix"
WRITE_SPEECH_ACT_TAG_FILENAME = "../Data/Collection_Speech_Act_Tagged"


tweet_enricher = TweetEnricher()
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
            collection_tweets[row[0]] = row[1]
            collection_urls[row[0]] = row[11]


speech_act_tags = tweet_enricher.speechActTagCollection(collection_tweets)

# speech act tagged collection
with open(WRITE_SPEECH_ACT_TAG_FILENAME, 'wb') as out_file:
    pickle.dump(speech_act_tags, out_file)

print("Collection - speech act tagged -"+ str(datetime.datetime.now().time()))

#When reading pre-tagged collection
SA_tagged_collection = pickle.load(open(WRITE_SPEECH_ACT_TAG_FILENAME, "rb"))

#create n-gram count matrix and get list of features(basic and with n grams)
basic_tweet_features,tweet_features,collection_n_gram_count_matrix = tweet_enricher.createNGramCountMatrix(collection,SA_tagged_collection, True)

# collection n-gram count matrix(uni,bi and tri grams)
# with open(WRITE_N_GRAM_MATRIX_FILENAME, 'wb') as out_file:
#     pickle.dump(collection_n_gram_count_matrix,out_file)
#
# print("Collection's n gram matrix generated -"+  str(datetime.datetime.now().time()))
# collection_n_gram_count_matrix_from_pickle = pickle.load(open(WRITE_N_GRAM_MATRIX_FILENAME, "rb"))

print("Best unigrams, bigrams and tri-grams chosen from collection and written in file." + str(datetime.datetime.now().time()))
