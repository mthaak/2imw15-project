import sklearn
import TweetEnricher
import pandas as pd
import os

################################################
# LOAD INPUT
################################################

from DataCollection.utils import read_csv

df = read_csv(os.path.join(os.pardir, 'DataCollection', 'results', 'search_20161024_004952_tweets.csv'),
              index_col='tweet_id')

################################################
# EXTRACT FEATURES
################################################

# CODE

################################################
# TRAIN CLASSIFIER
################################################

# CODE

################################################
# EVALUATE CLASSIFIER
################################################

# CODE

################################################
# OUTPUT
################################################
