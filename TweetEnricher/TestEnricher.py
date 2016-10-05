import csv
from tweet_Enricher import *

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

        #remove stop words
        tokens = tweet_enricher.removeStopWords(tokens)

        #add negative opinion feature
        row.append(tweet_enricher.hasNegativeOpinions(tokens))

        #add positive opinions feature
        row.append(tweet_enricher.hasPositiveOpinions(tokens))

        #add  Vulgar words feature
        row.append(tweet_enricher.hasVulgarWords(tokens))

        #add Emoticons feature
        row.append(tweet_enricher.hasEmoticons(tokens))

        #add Interrogation feature
        row.append(tweet_enricher.isInterrogative(tokens))

        #add Exclamation feature
        row.append((tweet_enricher.isExclamatory(tokens)))

        #add Abbreviations feature
        row.append(tweet_enricher.hasAbbreviations(tokens))

        #add twitter jargons feature
        row.append(tweet_enricher.hasTwitterJargons(tokens))

        #add Twiiter specific characters' features- presence and position
        presence, position = tweet_enricher.hasHash(tokens)
        row.append(presence)
        row.append(position)

        presence, position = tweet_enricher.hasATag(tokens)
        row.append(presence)
        row.append(position)

        presence, position = tweet_enricher.hasRT(tokens)
        row.append(presence)
        row.append(position)

        #add URL presence feature
        row.append(tweet_enricher.hasALink(tokens))

        enriched_tweets.append(row)

# Write enriched tweets to file
with open(WRITE_FILENAME, 'w', encoding='utf-8', newline='') as csv_out_file:
    writer = csv.writer(csv_out_file, delimiter='\t')
    for tweet in enriched_tweets:
        writer.writerow(tweet)

