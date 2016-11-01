import os
from TweetEnricher.tweetEnricher import TweetEnricher
import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score


def extract_features(file_name):
    """
    Extract features from tweets data and save them to pickle.
    Data is loaded from ../DataCollection/results folder.
    :param file_name: Name of CSV file to load tweets data from.
    :return: features dataFrame
    """
    assert isinstance(file_name, str) and len(file_name) > 0

    from DataCollection.utils import read_csv_ignore_comments as read_csv
    df = read_csv(os.path.join(os.pardir, 'DataCollection', 'results', file_name))

    # Load the tweet enricher
    tweet_enricher = TweetEnricher()
    tokens = df['text'].apply(tweet_enricher.tokenize).apply(tweet_enricher.removeStopWords)

    # Load features
    features = df['tweet_id'].to_frame()
    features['#followers'] = df['#followers']
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
    features['has#'] = tokens.apply(tweet_enricher.hasHash)
    features['#Position'] = features['has#'].apply(lambda x: x[1])
    features['has#'] = features['has#'].apply(lambda x: x[0])
    features['hasRT'] = tokens.apply(tweet_enricher.hasRT)
    features['RTPosition'] = features['hasRT'].apply(lambda x: x[1])
    features['hasRT'] = features['hasRT'].apply(lambda x: x[0])
    features['has@'] = tokens.apply(tweet_enricher.hasATag)
    features['@Position'] = features['has@'].apply(lambda x: x[1])
    features['has@'] = features['has@'].apply(lambda x: x[0])
    # features['hasNegativeOpinions'] = tokens.apply(tweet_enricher.hasNegativeOpinions).apply(lambda x: 1 if x[1] else 0)
    # features['hasPositiveOpinions'] = tokens.apply(tweet_enricher.hasPositiveOpinions).apply(lambda x: 1 if x[1] else 0)
    features['isRumor'] = pd.Series(0, index=np.arange(features.shape[0]))

    # Set tweet_id as the index for the features matrix
    features = features.set_index('tweet_id')
    print(features.head(2))

    pickle.dump(features, open(os.path.splitext(file_name)[0] + '_features.p', "wb"))
    return features


def classify(features):
    """
    Train and evaluates a classifier.
    :param features:
    :return: learned classifier model
    """
    X = features[features.columns.pop('isRumor')].as_matrix()
    y = features['isRumor'].as_matrix()
    clf = MultinomialNB
    GridSearchCV(clf, param_grid={}, cv=5, scoring='f1')


if __name__ == "__main__":
    # EXTRACT FEATURES
    file_name = 'search_20161024_004952_tweets.csv'
    df = extract_features(file_name)
    # df = pickle.load(open(os.path.splitext(file_name)[0] + '_features.p', "rb"))

    # TRAIN CLASSIFIER
    classify(features=df)
