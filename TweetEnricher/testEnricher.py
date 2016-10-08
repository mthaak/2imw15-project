import csv

from TweetEnricher.tweetEnricher import TweetEnricher
from tweetEnricher import *

READ_FILENAME="../Data/tweets.csv"
WRITE_FILENAME="../Data/enrichedTweets.csv"

tweet_enricher = TweetEnricher()
tokens=[] #hold tokens
enriched_tweets = [] # tweet with features

# Read tweet data from file
with open(READ_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):

        #tokenize the tweet text
        tokens = tweet_enricher.tokenize(row[3])

        # enrich tweets and store in list
        enriched_tweets.append(tweet_enricher.enrichTweets(row,tokens))

# Write enriched tweets to file
with open(WRITE_FILENAME, 'w', encoding='utf-8', newline='') as csv_out_file:
    writer = csv.writer(csv_out_file, delimiter='\t')
    for tweet in enriched_tweets:
        writer.writerow(tweet)

