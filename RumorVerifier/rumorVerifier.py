import os
from TweetEnricher.tweetEnricher import TweetEnricher
import numpy as np
import pandas as pd
import math
from dateutil import parser
import ast
import pickle
from DataCollection.utils import read_csv_ignore_comments as read_csv
from RumorVerifier.utils import *
from sklearn.grid_search import GridSearchCV


def extract_features(file_name, coe_backup_file=''):
    """
    Extract features from tweets data and save them to pickle.
    Data is loaded from ../DataCollection/results folder.
    :param coe_backup_file:
    :param file_name: tweets data CSV file name.
    :return: features dataFrame
    """
    assert isinstance(file_name, str) and len(file_name) > 0
    assert isinstance(coe_backup_file, str)

    # read tweets data
    df = read_csv(os.path.join(os.pardir, 'DataCollection', 'results', file_name))  # , index_col='tweet_id')

    # initialize the tweet enricher
    tweet_enricher = TweetEnricher()
    tokens = df['text'].apply(tweet_enricher.tokenize).apply(tweet_enricher.removeStopWords)

    # initialize the twitter API
    cur_dir = os.path.abspath(os.path.curdir)
    os.chdir(os.path.join(os.path.pardir, 'DataCollection'))
    from DataCollection.twitter_api import get_tweets_of_user, search_tweets
    from DataCollection.utils import days_delta
    os.chdir(cur_dir)

    # load features dataframe
    features = df['tweet_id'].to_frame()
    # if not coe_backup_file:
    #     features = df['tweet_id'].to_frame()
    # else:
    if coe_backup_file:
        a = pickle.load(open(coe_backup_file, 'rb'))
        for k in a.columns:
            features[k] = a[k]
        df = df.head(features.shape[0])

    # set tweet_id as the index for the features matrix
    features = features.set_index('tweet_id')

    try:
        if not coe_backup_file:
            # store the list of users for whom the below features is already computed
            # to prevent recomputing
            users = {}

            print('...extracting controversiality, originality, engagement')
            from progressbar import ProgressBar, Counter, ETA, Percentage
            widgets = ['> Processed: ', Counter(), ' (', Percentage(), ') ', ETA()]
            bar = ProgressBar(widgets=widgets, max_value=df.shape[0], redirect_stdout=True).start()
            for i, (tweet_id, row) in bar(enumerate(df.iterrows())):
                if row['screen_name'] not in users:
                    col, tweets = get_tweets_of_user(row['screen_name'], count=1000, save_to_csv=False)
                    tweets = pd.DataFrame(tweets, columns=col).set_index('tweet_id')

                    # CONTROVERSIALITY OF USER
                    col, replies = search_tweets('@' + row['screen_name'], count=500, save_to_csv=False)
                    replies = pd.DataFrame(replies, columns=col).set_index('tweet_id')
                    p_count, n_count = 0, 0
                    for id, r in tweets.iterrows():
                        id = int(id)
                        replies_to_id = replies.loc[replies['reply_to_tweet_id'] == id]
                        replies_to_id = replies_to_id['text'].apply(tweet_enricher.sentiment) \
                            .apply(lambda x: None if x[0] == x[1] else x[0] > x[1]).value_counts()
                        p_count += replies_to_id[True] if True in replies_to_id.index.values else 0
                        n_count += replies_to_id[False] if False in replies_to_id.index.values else 0
                    controversiality = math.pow(p_count + n_count,
                                                min(p_count / (n_count + 1), n_count / (p_count + 1))) \
                        if p_count + n_count > 0 else 0
                    features.set_value(tweet_id, 'controversiality', controversiality)

                    # ORIGINALITY OF USER
                    is_rt = tweets['text'].apply(tweet_enricher.tokenize)
                    is_rt = is_rt.apply(tweet_enricher.hasRT).apply(lambda x: x[0] == 1).value_counts()
                    originality = (is_rt[False] if False in is_rt.index.values else 0) / \
                                  (is_rt[True] if True in is_rt.index.values else 1)
                    features.set_value(tweet_id, 'originality', originality)

                    # ENGAGEMENT OF USER
                    user_created_at = parser.parse(row['user_created_at'])
                    engagement = (row['#statuses'] + row['#favourites']) / (days_delta(user_created_at,
                                                                                       user_created_at.today()) + 1)
                    features.set_value(tweet_id, 'engagement', engagement)

                    users[row['screen_name']] = controversiality, originality, engagement
                else:
                    features.set_value(tweet_id, 'controversiality', users[row['screen_name']][0])
                    features.set_value(tweet_id, 'originality', users[row['screen_name']][1])
                    features.set_value(tweet_id, 'engagement', users[row['screen_name']][2])
                bar.update(i)

                # after every 100 users, backup the data collected so far
                if i > 0 and i % 100 == 0:
                    pickle.dump(features, open('features_backup_(%s_processed).p' % i, 'wb'))
            bar.finish()

        print('...extracting credibility')
        print(df['verified'].value_counts())
        features['credibility'] = df['verified']

        print('...extracting influence')
        features['influence'] = df['#followers']

        print('...extracting role')
        df['#followings'] = df['#followings'].replace(0, 1)
        features['role'] = df.apply(lambda x: x['#followers'] / x['#followings'], axis=1)

        print('...extracting hasVulgarWords')
        features['hasVulgarWords'] = tokens.apply(tweet_enricher.hasVulgarWords)

        print('...extracting hasEmoticons')
        features['hasEmoticons'] = tokens.apply(tweet_enricher.hasEmoticons)

        print('...extracting isInterrogative')
        features['isInterrogative'] = tokens.apply(tweet_enricher.isInterrogative)

        print('...extracting isExclamatory')
        features['isExclamatory'] = tokens.apply(tweet_enricher.isExclamatory)

        print('...extracting hasAbbreviations')
        features['hasAbbreviations'] = tokens.apply(tweet_enricher.hasAbbreviations)

        print('...extracting hasTwitterJargons')
        features['hasTwitterJargons'] = tokens.apply(tweet_enricher.hasTwitterJargons)

        print('...extracting hasFPP')
        features['hasFPP'] = tokens.apply(tweet_enricher.hasFirstPersonPronouns)

        print('...extracting hasSource, nr_of_sources')
        urls = df['urls'].apply(ast.literal_eval)
        features['hasSource'] = urls.apply(lambda x: 1 if len(x) > 0 else 0)
        features['nr_of_sources'] = urls.apply(lambda x: len(x))

        # print('...extracting hasSpeechActVerbs')
        # hasSpeechActVerbs = tokens.apply(tweet_enricher.hasSpeechActVerbs)
        # for key in tweet_enricher.speech_act_verbs:
        #     features[key] = hasSpeechActVerbs.apply(lambda x: x[key])

        print('...extracting has#, #Position')
        has_hash = tokens.apply(tweet_enricher.hasHash)
        features['has#'] = has_hash.apply(lambda x: x[0])
        features['#Position'] = has_hash.apply(lambda x: x[1])

        print('...extracting hasRT, RTPosition')
        has_RT = tokens.apply(tweet_enricher.hasRT)
        features['hasRT'] = has_RT.apply(lambda x: x[0])
        features['RTPosition'] = has_RT.apply(lambda x: x[1])

        print('...extracting has@, @Position')
        has_a_tag = tokens.apply(tweet_enricher.hasATag)
        features['has@'] = has_a_tag.apply(lambda x: x[0])
        features['@Position'] = has_a_tag.apply(lambda x: x[1])

        # print('...extracting hasNegativeOpinions, hasPositiveOpinions')
        # features['hasNegativeOpinions'] = tokens.apply(tweet_enricher.hasNegativeOpinions).apply(lambda x: 1 if x[1] else 0)
        # features['hasPositiveOpinions'] = tokens.apply(tweet_enricher.hasPositiveOpinions).apply(lambda x: 1 if x[1] else 0)

        # print('...creating isRumor column for manual labelling')
        # features['isRumor'] = pd.Series(0, index=np.arange(features.shape[0]))
    except Exception as e:
        print(e)
        pickle.dump(features, open('recovered_features_dataframe.p', 'wb'))
    print(features.head(2))

    features.to_csv(os.path.join('results', os.path.splitext(file_name)[0] + '_features.csv'))
    pickle.dump(features, open(os.path.join('results', os.path.splitext(file_name)[0] + '_features.p'), "wb"))
    print('Done!')

    return features


def extract_cluster_features(file_name, features, feature_type='Gaussian', clusters=[]):
    assert isinstance(file_name, str) and len(file_name) > 0
    assert isinstance(features, pd.DataFrame) and 'tweet_id' in features.index
    assert isinstance(feature_type, str)
    assert isinstance(clusters, list)
    assert all(isinstance(x, list) and all(isinstance(y, int) and y > 0 for y in x) for x in clusters)

    # read tweets data
    df = read_csv(os.path.join(os.pardir, 'DataCollection', 'results', file_name))

    # initialize dataframe for storing cluster features
    cl_features = pd.DataFrame()

    if feature_type == 'Gaussian' or feature_type == 'G':
        for index, cluster in enumerate(clusters):
            df = df.loc[cluster]
            features = features.loc[cluster]

            # save cluster
            # df.to_csv(os.path.join('results', os.path.splitext(file_name)[0], 'tweets_cluster_%s.csv' % index))

            # number of tweets and unique users
            cl_features.set_value(index, '#tweets', features.shape[0])
            cl_features.set_value(index, '#users', len(df['screen_name'].unique()))

            # find count of low, medium and high controversial users
            features['controversiality'] = normalize(features['controversiality'])
            bins = (0, 0.05, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('controversialityLow', 'controversialityMedium', 'controversialityHigh')
            features['controversiality'] = discretize(features['controversiality'], bins, group_names)
            features = one_hot_encode(features, 'controversiality')
            for k in group_names:
                cl_features.set_value(index, k + 'Mean', features[k].mean())
                cl_features.set_value(index, k + 'Std', features[k].std())

            # find count of low, medium and high original users
            features['originality'] = normalize(features['originality'])
            bins = (0, 0.2, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('originalityLow', 'originalityMedium', 'originalityHigh')
            features['originality'] = discretize(features['originality'], bins, group_names)
            features = one_hot_encode(features, 'originality')
            for k in group_names:
                cl_features.set_value(index, k + 'Mean', features[k].mean())
                cl_features.set_value(index, k + 'Std', features[k].std())

            # find count of low, medium and high active users
            features['engagement'] = normalize(features['engagement'])
            bins = (0, 0.2, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('engagementLow', 'engagementMedium', 'engagementHigh')
            features['engagement'] = discretize(features['engagement'], bins, group_names)
            features = one_hot_encode(features, 'engagement')
            for k in group_names:
                cl_features.set_value(index, k + 'Mean', features[k].mean())
                cl_features.set_value(index, k + 'Std', features[k].std())

            # find count of low, medium and high reach users
            features['influence'] = normalize(features['influence'])
            bins = (0, 0.3, 0.7, 1)  # TODO: Pick good binning values
            group_names = ('influenceLow', 'influenceMedium', 'influenceHigh')
            features['influence'] = discretize(features['influence'], bins, group_names)
            features = one_hot_encode(features, 'influence')
            for k in group_names:
                cl_features.set_value(index, k + 'Mean', features[k].mean())
                cl_features.set_value(index, k + 'Std', features[k].std())

            # find count of low, medium and high influence users
            features['role'] = normalize(features['role'])
            bins = (0, 0.2, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('roleLow', 'roleMedium', 'roleHigh')
            features['role'] = discretize(features['role'], bins, group_names)
            features = one_hot_encode(features, 'role')
            for k in group_names:
                cl_features.set_value(index, k + 'Mean', features[k].mean())
                cl_features.set_value(index, k + 'Std', features[k].std())

            # find the number of credible users
            df1 = features[features['credibility'] == 1].join(df)
            credibility = df1.drop_duplicates(subset='screen_name')['credibility']
            df1 = None
            cl_features.set_value(index, 'credibilityMean', credibility.mean())
            cl_features.set_value(index, '#credibleUsers', credibility.sum())

            # other linguistic features
            cl_features.set_value(index, 'hasVulgarWords', features['hasVulgarWords'].mean())
            cl_features.set_value(index, 'hasEmoticons', features['hasEmoticons'].mean())
            cl_features.set_value(index, 'isInterrogative', features['isInterrogative'].mean())
            cl_features.set_value(index, 'isExclamatory', features['isExclamatory'].mean())
            cl_features.set_value(index, 'hasAbbreviations', features['hasAbbreviations'].mean())
            cl_features.set_value(index, 'hasTwitterJargons', features['hasTwitterJargons'].mean())
            cl_features.set_value(index, 'hasFPP', features['hasFPP'].mean())
            cl_features.set_value(index, 'hasSource', features['hasSource'].mean())
            cl_features.set_value(index, 'nr_of_sources', features['nr_of_sources'].mean())
            # for key in tweet_enricher.speech_act_verbs:
            #     cl_df[index, key] = df[key].mean()
            cl_features.set_value(index, 'has#', features['has#'].mean())
            cl_features.set_value(index, '#Position', features['#Position'].mean())
            cl_features.set_value(index, 'hasRT', features['hasRT'].mean())
            cl_features.set_value(index, 'RTPosition', features['RTPosition'].mean())
            cl_features.set_value(index, 'has@', features['has@'].mean())
            cl_features.set_value(index, '@Position', features['@Position'].mean())
            cl_features.set_value(index, 'isRumor', 0)
    elif feature_type == 'Multivariate' or feature_type == 'Mv':
        for index, cluster in enumerate(clusters):
            df = df.loc[cluster]
            features = features.loc[cluster]

            # save clusters
            # df.to_csv(os.path.join('results', os.path.splitext(file_name)[0], 'tweets_cluster_%s.csv' % index))

            # number of tweets and unique users
            cl_features.set_value(index, '#tweets', features.shape[0])
            cl_features.set_value(index, '#users', len(df['screen_name'].unique()))

            # find count of low, medium and high controversial users
            cl_features.set_value(index, 'controversialityLow', features['controversiality'].value_counts())
            cl_features.set_value(index, 'controversialityMedium', features['controversiality'].mean())
            cl_features.set_value(index, 'controversialityHigh', features['controversiality'].mean())

            # find count of low, medium and high original users
            cl_features.set_value(index, 'originality_mean', features['originality'].mean())
            cl_features.set_value(index, 'originality_mean', features['originality'].mean())
            cl_features.set_value(index, 'originality_mean', features['originality'].mean())

            # find count of low, medium and high active users
            cl_features.set_value(index, 'engagement_mean', features['engagement'].mean())
            cl_features.set_value(index, 'engagement_mean', features['engagement'].mean())
            cl_features.set_value(index, 'engagement_mean', features['engagement'].mean())

            # find count of low, medium and high reach users
            cl_features.set_value(index, 'influence_mean', features['influence'].mean())
            cl_features.set_value(index, 'influence_mean', features['influence'].mean())
            cl_features.set_value(index, 'influence_mean', features['influence'].mean())

            # find count of low, medium and high influence users
            cl_features.set_value(index, 'role_mean', features['role'].mean())
            cl_features.set_value(index, 'role_mean', features['role'].mean())
            cl_features.set_value(index, 'role_mean', features['role'].mean())

            # find the number of credible users
            cl_features.set_value(index, 'credibility', features[features['screen_name'].unique()].sum())
            cl_features.set_value(index, 'hasVulgarWords', features['hasVulgarWords'].mean())
            cl_features.set_value(index, 'hasEmoticons', features['hasEmoticons'].mean())
            cl_features.set_value(index, 'isInterrogative', features['isInterrogative'].mean())
            cl_features.set_value(index, 'isExclamatory', features['isExclamatory'].mean())
            cl_features.set_value(index, 'hasAbbreviations', features['hasAbbreviations'].mean())
            cl_features.set_value(index, 'hasTwitterJargons', features['hasTwitterJargons'].mean())
            cl_features.set_value(index, 'hasFPP', features['hasFPP'].mean())
            cl_features.set_value(index, 'hasSource', features['hasSource'].mean())
            cl_features.set_value(index, 'nr_of_sources', features['nr_of_sources'].mean())
            # for key in tweet_enricher.speech_act_verbs:
            #     cl_df[index, key] = df[key].mean()
            cl_features.set_value(index, 'has#', features['has#'].mean())
            cl_features.set_value(index, '#Position', features['#Position'].mean())
            cl_features.set_value(index, 'hasRT', features['hasRT'].mean())
            cl_features.set_value(index, 'RTPosition', features['RTPosition'].mean())
            cl_features.set_value(index, 'has@', features['has@'].mean())
            cl_features.set_value(index, '@Position', features['@Position'].mean())
            cl_features.set_value(index, 'isRumor', 0)
    elif feature_type == 'Multinomial' or feature_type == 'Mn':
        for index, cluster in enumerate(clusters):
            df = df.loc[cluster]
            features = features.loc[cluster]

            # save clusters
            # features.to_csv(os.path.join('results', os.path.splitext(file_name)[0], 'tweets_cluster_%s.csv' % index))

            cl_features.set_value(index, '#tweets', features.shape[0])
            cl_features.set_value(index, '#users', len(df['screen_name'].unique()))
            cl_features.set_value(index, 'controversiality', features['controversiality'].sum())
            cl_features.set_value(index, 'originality_mean', features['originality'].mean())
            cl_features.set_value(index, 'originality_std', features['originality'].std())
            cl_features.set_value(index, 'engagement_mean', features['engagement'].mean())
            cl_features.set_value(index, 'engagement_std', features['engagement'].std())
            cl_features.set_value(index, 'influence_mean', features['influence'].mean())
            cl_features.set_value(index, 'influence_std', features['influence'].std())
            cl_features.set_value(index, 'role_mean', features['role'].mean())
            cl_features.set_value(index, 'role_std', features['role'].std())
            cl_features.set_value(index, 'credibility', features['credibility'].mean())
            cl_features.set_value(index, 'hasVulgarWords', features['hasVulgarWords'].mean())
            cl_features.set_value(index, 'hasEmoticons', features['hasEmoticons'].mean())
            cl_features.set_value(index, 'isInterrogative', features['isInterrogative'].mean())
            cl_features.set_value(index, 'isExclamatory', features['isExclamatory'].mean())
            cl_features.set_value(index, 'hasAbbreviations', features['hasAbbreviations'].mean())
            cl_features.set_value(index, 'hasTwitterJargons', features['hasTwitterJargons'].mean())
            cl_features.set_value(index, 'hasFPP', features['hasFPP'].mean())
            cl_features.set_value(index, 'hasSource', features['hasSource'].mean())
            cl_features.set_value(index, 'nr_of_sources', features['nr_of_sources'].mean())
            # for key in tweet_enricher.speech_act_verbs:
            #     cl_df[index, key] = df[key].mean()
            cl_features.set_value(index, 'has#', features['has#'].mean())
            cl_features.set_value(index, '#Position', features['#Position'].mean())
            cl_features.set_value(index, 'hasRT', features['hasRT'].mean())
            cl_features.set_value(index, 'RTPosition', features['RTPosition'].mean())
            cl_features.set_value(index, 'has@', features['has@'].mean())
            cl_features.set_value(index, '@Position', features['@Position'].mean())
            cl_features.set_value(index, 'isRumor', 0)

        # save features
        cl_features.to_csv(os.path.join('results', os.path.splitext(file_name)[0], 'cluster_features.csv'))
    else:
        raise ValueError('Feature type can only be Gaussian(G)/ Multinomial(Mn)/ Multivariate(Mv).')


def classify(df, clf, param_grid={}):
    """
    Train and evaluates a classifier.
    :param df:
    :param clf:
    :param param_grid:
    :return: learned classifier model
    """
    assert isinstance(df, pd.DataFrame) and 'tweet_id' in df.index
    assert isinstance(param_grid, dict)
    assert clf is not None

    # load the data
    X = df[df.columns.pop('isRumor')].as_matrix()
    y = df['isRumor'].as_matrix()

    # instantiate and train the model
    grid = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X, y)

    # view the complete results
    print(grid.grid_scores_)

    # examine the best model
    print(grid.best_score_)
    print(grid.best_params_)
