import os
from TweetEnricher.tweetEnricher import TweetEnricher
import numpy as np
import pandas as pd
import math
from dateutil import parser
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
    features['credibility'] = df['verified']
    features['influence'] = df['#followers']
    features['role'] = df.apply(lambda x: x['#followers'] / x['#followings'])

    from DataCollection.twitter_api import get_tweets_of_user, search_tweets
    from DataCollection.utils import days_delta

    # store the list of users for whom the below features is already computed
    # to prevent recomputing
    users = {}

    for index, row in df.iterrows():
        if row['screen_name'] not in users:
            col, tweets = get_tweets_of_user(row['screen_name'], nr_of_tweets=1000, save_to_csv=False)
            tweets = pd.DataFrame(tweets, columns=col).set_index('tweet_id')

            # CONTROVERSIALITY OF USER
            col, replies = search_tweets('@' + row['screen_name'], nr_of_tweets=1000, save_to_csv=False)
            replies = pd.DataFrame(replies, columns=col).set_index('tweet_id')
            p_count, n_count = 0, 0
            for i, r in tweets:
                replies_to_i = replies[replies['reply_to_tweet_id'] == i]
                replies_to_i = replies_to_i['text'].apply(tweet_enricher.sentiment) \
                    .apply(lambda x: None if x[0] == x[1] else x[0] > x[1]).value_counts()
                p_count += replies_to_i[True]
                n_count += replies_to_i[False]
            controversiality = math.pow(p_count + n_count, min(p_count / (n_count + 1), n_count / (p_count + 1)))
            df.loc[index, 'controversiality'] = controversiality

            # ORIGINALITY OF USER
            is_rt = tweets['text'].apply(tweet_enricher.tokenize)
            is_rt = is_rt.apply(tweet_enricher.hasRT).apply(lambda x: x[0] == 1).value_counts()
            originality = is_rt[False] / is_rt[True]
            df.loc[index, 'originality'] = originality

            # ENGAGEMENT OF USER
            user_created_at = parser.parse(row['user_created_at'])
            engagement = (row['#statuses'] + row['#favourites']) / days_delta(user_created_at, user_created_at.today())
            df.loc[index, 'engagement'] = engagement

            users[row['screen_name']] = controversiality, originality, engagement
        else:
            df.loc[index, 'controversiality'] = users[row['screen_name']][0]
            df.loc[index, 'originality'] = users[row['screen_name']][1]
            df.loc[index, 'engagement'] = users[row['screen_name']][2]

    features['hasVulgarWords'] = tokens.apply(tweet_enricher.hasVulgarWords)
    features['hasEmoticons'] = tokens.apply(tweet_enricher.hasEmoticons)
    features['isInterrogative'] = tokens.apply(tweet_enricher.isInterrogative)
    features['isExclamatory'] = tokens.apply(tweet_enricher.isExclamatory)
    features['hasAbbreviations'] = tokens.apply(tweet_enricher.hasAbbreviations)
    features['hasTwitterJargons'] = tokens.apply(tweet_enricher.hasTwitterJargons)
    features['hasFPP'] = tokens.apply(tweet_enricher.hasFirstPersonPronouns)
    features['hasSource'] = df['urls'].apply(lambda x: 1 if len(x) > 0 else 0)

    hasSpeechActVerbs = tokens.apply(tweet_enricher.hasSpeechActVerbs)
    for key in tweet_enricher.speech_act_verbs:
        features[key] = hasSpeechActVerbs.apply(lambda x: x[key])

    has_hash = tokens.apply(tweet_enricher.hasHash)
    features['has#'] = has_hash.apply(lambda x: x[0])
    features['#Position'] = has_hash.apply(lambda x: x[1])

    has_RT = tokens.apply(tweet_enricher.hasRT)
    features['hasRT'] = has_RT.apply(lambda x: x[0])
    features['RTPosition'] = has_RT.apply(lambda x: x[1])

    has_a_tag = tokens.apply(tweet_enricher.hasATag)
    features['has@'] = has_a_tag.apply(lambda x: x[0])
    features['@Position'] = has_a_tag.apply(lambda x: x[1])

    # features['hasNegativeOpinions'] = tokens.apply(tweet_enricher.hasNegativeOpinions).apply(lambda x: 1 if x[1] else 0)
    # features['hasPositiveOpinions'] = tokens.apply(tweet_enricher.hasPositiveOpinions).apply(lambda x: 1 if x[1] else 0)

    features['isRumor'] = pd.Series(0, index=np.arange(features.shape[0]))

    # Set tweet_id as the index for the features matrix
    features = features.set_index('tweet_id')
    print(features.head(2))

    # features.to_csv(os.path.join('results', os.path.splitext(file_name)[0] + '_features.csv'))
    pickle.dump(features, open(os.path.join('results', os.path.splitext(file_name)[0] + '_features.p'), "wb"))
    return features


def extract_cluster_features(file_name, features, clusters=[]):
    assert isinstance(clusters, list)
    assert all(isinstance(x, list) and all(isinstance(y, int) and y > 0 for y in x) for x in clusters)

    # Load the tweet enricher
    tweet_enricher = TweetEnricher()

    cl_df = pd.DataFrame()
    for index, cluster in enumerate(clusters):
        df = features.loc[cluster]
        df.to_csv(os.path.join('results', os.path.splitext(file_name)[0] + '_features_cluster_%s.csv' % index))
        cl_df[index, 'credibility'] = df['credibility'].mean()
        cl_df[index, 'influence'] = df['influence'].mean()
        cl_df[index, 'role'] = df['role'].mean()
        cl_df[index, 'controversiality'] = df['controversiality'].mean()
        cl_df[index, 'originality'] = df['originality'].mean()
        cl_df[index, 'engagement'] = df['engagement'].mean()
        cl_df[index, 'hasVulgarWords'] = df['hasVulgarWords'].mean()
        cl_df[index, 'hasEmoticons'] = df['hasEmoticons'].mean()
        cl_df[index, 'isInterrogative'] = df['isInterrogative'].mean()
        cl_df[index, 'isExclamatory'] = df['isExclamatory'].mean()
        cl_df[index, 'hasAbbreviations'] = df['hasAbbreviations'].mean()
        cl_df[index, 'hasTwitterJargons'] = df['hasTwitterJargons'].mean()
        cl_df[index, 'hasFPP'] = df['hasFPP'].mean()
        for key in tweet_enricher.speech_act_verbs:
            cl_df[index, key] = df[key].mean()
        cl_df[index, 'has#'] = df['has#'].mean()
        cl_df[index, '#Position'] = df['#Position'].mean()
        cl_df[index, 'hasRT'] = df['hasRT'].mean()
        cl_df[index, 'RTPosition'] = df['RTPosition'].mean()
        cl_df[index, 'has@'] = df['has@'].mean()
        cl_df[index, '@Position'] = df['@Position'].mean()
        cl_df[index, 'isRumor'] = df['isRumor'].mean()

        # pickle.dump(clusters, open(os.path.join('results', os.path.splitext(file_name)[0] + '_clusters.p')), "wb")


def classify(df):
    """
    Train and evaluates a classifier.
    :param df:
    :return: learned classifier model
    """
    X = df[df.columns.pop('isRumor')].as_matrix()
    y = df['isRumor'].as_matrix()
    clf = MultinomialNB
    GridSearchCV(clf, param_grid={}, cv=5, scoring='f1', n_jobs=-1)


if __name__ == "__main__":
    # EXTRACT FEATURES
    file_name = 'search_20161102_211623_tweets.csv'
    df = extract_features(file_name)
    # df = pickle.load(open(os.path.splitext(file_name)[0] + '_features.p', "rb"))

    # EXTRACT CLUSTER FEATURES
    # clusters = pickle.load(open(os.path.join('results', os.path.splitext(file_name)[0] + '_clusters.p')), "rb")
    # extract_cluster_features(file_name, df, clusters)

    # TRAIN CLASSIFIER
    # classify(features=df)
