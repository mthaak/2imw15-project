import csv
import pickle
import scipy.sparse as sp

from TweetEnricher.tweetEnricher import TweetEnricher
from tweetEnricher import *

READ_FILENAME="../Data/tweets.csv"
WRITE_FILENAME="../Data/features"

tweet_enricher = TweetEnricher()
tokens=[] #hold tokens
enriched_tweets = [] # tweet with features
document=[] # stores a document of all the tweets
# Read tweet data from file
with open(READ_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):
        #build document for n-grmas count
        document.append(row[3])

#create n-gram count matrix and get list of features
tweet_features = tweet_enricher.createNGramCountMatrix(document)

for tweet in document:
    # enrich tweets and store in list
    enriched_tweets.append(tweet_enricher.enrichTweets(tweet))


#tweet-feature matrix pickeld
with open(WRITE_FILENAME, 'wb') as out_file:
    pickle.dump(sp.csc_matrix(enriched_tweets),out_file)

#Write enriched tweets to file
'''with open(WRITE_FILENAME, 'w', encoding='utf-8', newline='') as csv_out_file:
    writer = csv.writer(csv_out_file, delimiter=',')
    writer.writerow(tweet_features)
    for tweet in enriched_tweets:
        writer.writerow(tweet)'''

