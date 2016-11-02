import sklearn
from TweetEnricher.tweetEnricher import TweetEnricher
import os

################################################
# LOAD INPUT
################################################

from DataCollection.utils import read_csv_ignore_comments as read_csv

df = read_csv(os.path.join(os.pardir, 'DataCollection', 'results', 'search_20161024_004952_tweets.csv'))

################################################
# EXTRACT FEATURES
################################################

tweet_enricher = TweetEnricher()
tokens = df['text'].apply(tweet_enricher.tokenize).apply(tweet_enricher.removeStopWords)

# Load features
features = df['tweet_id'].to_frame()
features['hasVulgarWords'] = tokens.apply(tweet_enricher.hasVulgarWords)
features['hasEmoticons'] = tokens.apply(tweet_enricher.hasEmoticons)
features['isInterrogative'] = tokens.apply(tweet_enricher.isInterrogative)
features['isExclamatory'] = tokens.apply(tweet_enricher.isExclamatory)
features['hasAbbreviations'] = tokens.apply(tweet_enricher.hasAbbreviations)
features['hasTwitterJargons'] = tokens.apply(tweet_enricher.hasTwitterJargons)
features['hasALink'] = df['urls'].apply(lambda x: 1 if len(x) > 0 else 0)

hasSpeechActVerbs = tokens.apply(tweet_enricher.hasSpeechActVerbs)
for key in tweet_enricher.speech_act_verbs:
    features[key] = hasSpeechActVerbs.apply(lambda x: x[key])
hasSpeechActVerbs = None

features['has#'] = tokens.apply(tweet_enricher.hasHash)
features['#Position'] = features['has#'].apply(lambda x: x[1])
features['has#'] = features['has#'].apply(lambda x: x[0])

features['hasRT'] = tokens.apply(tweet_enricher.hasRT)
features['RTPosition'] = features['hasRT'].apply(lambda x: x[1])
features['hasRT'] = features['hasRT'].apply(lambda x: x[0])

features['has@'] = tokens.apply(tweet_enricher.hasATag)
features['@Position'] = features['has@'].apply(lambda x: x[1])
features['has@'] = features['has@'].apply(lambda x: x[0])

features['hasNegativeOpinions'] = tokens.apply(tweet_enricher.hasNegativeOpinions).apply(lambda x: 1 if x[1] else 0)
features['hasPositiveOpinions'] = tokens.apply(tweet_enricher.hasPositiveOpinions).apply(lambda x: 1 if x[1] else 0)

features = features.set_index('tweet_id')
print(features.head(2))

################################################
# TRAIN CLASSIFIER
################################################



################################################
# EVALUATE CLASSIFIER
################################################

# CODE

################################################
# OUTPUT
################################################
