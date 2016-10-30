import sklearn
import TweetEnricher
import pandas as pd
import os

################################################
# LOAD INPUT
################################################

dataFile = open(os.path.join(os.pardir, 'DataCollection', 'results', 'search_20161024_004952_tweets.csv'),
                'r',
                encoding='utf-8')
df = pd.read_csv(dataFile, sep="\t", index_col=None, header=0)

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
