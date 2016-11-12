import os
from TweetEnricher.tweetEnricher import TweetEnricher
import numpy as np
import pandas as pd
import math
from dateutil import parser
import ast
import pickle
from progressbar import ProgressBar, Counter, Bar, Percentage, ETA
from RumorVerifier.utils import *
from sklearn.grid_search import GridSearchCV


def extract_features(tweets, features=None, save_file_name='tweets', save_to_csv=True, save_to_pickle=True):
    """
    Extract features from tweets data and save them to pickle.
    Data is loaded from ../DataCollection/results folder.
    :param tweets: tweets dataFrame.
    :param features: Controversiality/Originality/Engagement features backup dataFrame
    :param save_file_name:
    :param save_to_csv:
    :param save_to_pickle:
    :return: features dataFrame
    """
    assert isinstance(tweets, pd.DataFrame) and 'tweet_id' == tweets.index.name
    assert (isinstance(features, pd.DataFrame) and 'tweet_id' == tweets.index.name) or features is None
    assert isinstance(save_to_csv, bool)
    assert isinstance(save_to_pickle, bool)
    assert isinstance(save_file_name, str) and (len(save_file_name) > 0 if (save_to_csv or save_to_pickle) else False)

    # initialize the tweet enricher
    tweet_enricher = TweetEnricher()
    tokens = tweets['text'].apply(tweet_enricher.tokenize).apply(tweet_enricher.removeStopWords)

    if features is None:
        # load features dataFrame
        features = pd.DataFrame(index=tweets.index)
    else:
        # take subset of tweets corresponding to features
        tweets = tweets.loc[tweets.index]

    try:
        if features.empty:
            # initialize the twitter API
            cur_dir = os.path.abspath(os.path.curdir)
            os.chdir(os.path.join(os.path.pardir, 'DataCollection'))
            from DataCollection.twitter_api import get_tweets_of_user, search_tweets
            from DataCollection.utils import days_delta
            os.chdir(cur_dir)

            # initialize a progress bar
            widgets = ['> Processed: ', Counter(), ' (', Percentage(), ') ', ETA()]
            bar = ProgressBar(widgets=widgets, max_value=tweets.shape[0], redirect_stdout=True).start()

            # store the list of users for whom the below features is already computed
            # to prevent recomputing
            users = {}

            print('...extracting controversiality, originality, engagement')
            for i, (tweet_id, row) in bar(enumerate(tweets.iterrows())):
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
                    pickle.dump(features[features.notnull()], open('features_backup_(%s_processed).p' % i, 'wb'))
            bar.finish()

        print('...extracting credibility')
        features['credibility'] = tweets['verified']

        print('...extracting influence')
        features['influence'] = tweets['#followers']

        print('...extracting role')
        tweets.set_value(tweets['#followings'] == 0, '#followings', 1)
        features['role'] = tweets.apply(lambda x: x['#followers'] / x['#followings'], axis=1)

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
        urls = tweets['urls'].apply(ast.literal_eval)
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
    except Exception as e:
        print(e)
        pickle.dump(features, open('recovered_features.p', 'wb'))

    # check to make sure
    print(features.head(2))

    # save features to both csv and pickle
    save_file_path = os.path.join('results', save_file_name)
    if save_to_csv:
        features.to_csv(save_file_path + '_features.csv')
    if save_to_pickle:
        pickle.dump(features, open(save_file_path + '_features.p', "wb"))

    print('Done!')
    return features


def extract_cluster_features(tweets, features, clusters, feature_type='Gaussian', save_file_name='tweets'):
    assert isinstance(tweets, pd.DataFrame) and 'tweet_id' == tweets.index.name
    assert isinstance(features, pd.DataFrame) and 'tweet_id' == features.index.name
    assert isinstance(clusters, pd.DataFrame) \
           and 'center_id' == clusters.index.name and 'tweet_ids' in clusters.columns
    # assert all(isinstance(x, list) and all(isinstance(y, int) and y > 0 for y in x) for x in clusters)
    assert isinstance(feature_type, str)
    assert isinstance(save_file_name, str) and len(save_file_name) > 0

    # initialize dataFrame for storing cluster features
    cl_features = pd.DataFrame(index=clusters.index)

    # initialize a progress bar
    widgets = ['> Processed: ', Percentage(), Bar()]
    bar = ProgressBar(widgets=widgets, max_value=clusters.shape[0]).start()

    if feature_type == 'Gaussian' or feature_type == 'G':
        """ Features must be continuous random variables with normal distribution """

        for i, (center_id, cluster) in bar(enumerate(clusters.iterrows())):
            df1 = tweets.loc[cluster.tweet_ids]
            df2 = features.loc[cluster.tweet_ids]

            # save cluster
            # df.to_csv(os.path.join('results', os.path.splitext(file_name)[0], 'tweets_cluster_%s.csv' % index))

            # number of tweets and unique users
            cl_features.set_value(center_id, '#tweets', df1.shape[0])
            cl_features.set_value(center_id, '#users', df1['screen_name'].unique().shape[0])

            # find count of low, medium and high controversial users
            df2['controversiality'] = normalize(df2['controversiality'])
            bins = (-0.001, 0.05, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('controversialityLow', 'controversialityMedium', 'controversialityHigh')
            df2['controversiality'] = discretize(df2['controversiality'], bins, group_names)
            df2 = one_hot_encode(df2, 'controversiality')
            for k in group_names:
                cl_features.set_value(center_id, k + 'Mean', df2[k].mean())
                cl_features.set_value(center_id, k + 'Std', df2[k].std() if df2[k].size > 1 else 0)

            # find count of low, medium and high original users
            df2['originality'] = normalize(df2['originality'])
            bins = (-0.001, 0.2, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('originalityLow', 'originalityMedium', 'originalityHigh')
            df2['originality'] = discretize(df2['originality'], bins, group_names)
            df2 = one_hot_encode(df2, 'originality')
            for k in group_names:
                cl_features.set_value(center_id, k + 'Mean', df2[k].mean())
                cl_features.set_value(center_id, k + 'Std', df2[k].std() if df2[k].size > 1 else 0)

            # find count of low, medium and high active users
            df2['engagement'] = normalize(df2['engagement'])
            bins = (-0.001, 0.2, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('engagementLow', 'engagementMedium', 'engagementHigh')
            df2['engagement'] = discretize(df2['engagement'], bins, group_names)
            df2 = one_hot_encode(df2, 'engagement')
            for k in group_names:
                cl_features.set_value(center_id, k + 'Mean', df2[k].mean())
                cl_features.set_value(center_id, k + 'Std', df2[k].std() if df2[k].size > 1 else 0)

            # find count of low, medium and high reach users
            df2['influence'] = normalize(df2['influence'])
            bins = (-0.001, 0.3, 0.7, 1)  # TODO: Pick good binning values
            group_names = ('influenceLow', 'influenceMedium', 'influenceHigh')
            df2['influence'] = discretize(df2['influence'], bins, group_names)
            df2 = one_hot_encode(df2, 'influence')
            for k in group_names:
                cl_features.set_value(center_id, k + 'Mean', df2[k].mean())
                cl_features.set_value(center_id, k + 'Std', df2[k].std() if df2[k].size > 1 else 0)

            # find count of low, medium and high influence users
            df2['role'] = normalize(df2['role'])
            bins = (-0.001, 0.2, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('roleLow', 'roleMedium', 'roleHigh')
            df2['role'] = discretize(df2['role'], bins, group_names)
            df2 = one_hot_encode(df2, 'role')
            for k in group_names:
                cl_features.set_value(center_id, k + 'Mean', df2[k].mean())
                cl_features.set_value(center_id, k + 'Std', df2[k].std() if df2[k].size > 1 else 0)

            # find the number of credible users
            credibility = df2[df2['credibility'] == 1].join(df1)
            credibility = credibility.drop_duplicates(subset='screen_name')['credibility']
            cl_features.set_value(center_id, 'credibilityMean', credibility.mean() if credibility.size > 0 else 0)
            cl_features.set_value(center_id, '#credibleUsers', credibility.sum())

            # other linguistic features
            cl_features.set_value(center_id, 'hasVulgarWords', df2['hasVulgarWords'].mean())
            cl_features.set_value(center_id, 'hasEmoticons', df2['hasEmoticons'].mean())
            cl_features.set_value(center_id, 'isInterrogative', df2['isInterrogative'].mean())
            cl_features.set_value(center_id, 'isExclamatory', df2['isExclamatory'].mean())
            cl_features.set_value(center_id, 'hasAbbreviations', df2['hasAbbreviations'].mean())
            cl_features.set_value(center_id, 'hasTwitterJargons', df2['hasTwitterJargons'].mean())
            cl_features.set_value(center_id, 'hasFPP', df2['hasFPP'].mean())
            cl_features.set_value(center_id, 'hasSource', df2['hasSource'].mean())
            cl_features.set_value(center_id, 'nr_of_sources', df2['nr_of_sources'].mean())
            # for key in tweet_enricher.speech_act_verbs:
            #     cl_features.set_value(index, key, df2[k].mean())
            cl_features.set_value(center_id, 'has#', df2['has#'].mean())
            cl_features.set_value(center_id, '#Position', df2['#Position'].mean())
            cl_features.set_value(center_id, 'hasRT', df2['hasRT'].mean())
            cl_features.set_value(center_id, 'RTPosition', df2['RTPosition'].mean())
            cl_features.set_value(center_id, 'has@', df2['has@'].mean())
            cl_features.set_value(center_id, '@Position', df2['@Position'].mean())
            cl_features.set_value(center_id, 'isRumor', 0)

            # update progress
            bar.update(i)
    elif feature_type == 'Multivariate' or feature_type == 'Mv':
        """ Features must be binary """
        return NotImplemented
    elif feature_type == 'Multinomial' or feature_type == 'Mn':
        """ Features must counts/categories """

        for i, (center_id, cluster) in bar(enumerate(clusters.iteritems())):
            df1 = tweets.loc[cluster.tweet_ids]
            df2 = features.loc[cluster.tweet_ids]

            # save clusters
            # features.to_csv(os.path.join('results', os.path.splitext(file_name)[0], 'tweets_cluster_%s.csv' % index))

            # number of tweets and unique users
            cl_features.set_value(center_id, '#tweets', df1.shape[0])
            cl_features.set_value(center_id, '#users', df1['screen_name'].unique().shape[0])

            # find count of low, medium and high controversial users
            df2['controversiality'] = normalize(df2['controversiality'])
            bins = (-0.001, 0.05, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('controversialityLow', 'controversialityMedium', 'controversialityHigh')
            df2['controversiality'] = discretize(df2['controversiality'], bins, group_names)
            df2 = one_hot_encode(df2, 'controversiality')
            for k in group_names:
                cl_features.set_value(center_id, k, df2[k].mean() > 0.5)  # TODO: Pick good cut-off value

            # find count of low, medium and high original users
            df2['originality'] = normalize(df2['originality'])
            bins = (-0.001, 0.2, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('originalityLow', 'originalityMedium', 'originalityHigh')
            df2['originality'] = discretize(df2['originality'], bins, group_names)
            df2 = one_hot_encode(df2, 'originality')
            for k in group_names:
                cl_features.set_value(center_id, k, df2[k].mean() > 0.5)  # TODO: Pick good cut-off value

            # find count of low, medium and high active users
            df2['engagement'] = normalize(df2['engagement'])
            bins = (-0.001, 0.2, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('engagementLow', 'engagementMedium', 'engagementHigh')
            df2['engagement'] = discretize(df2['engagement'], bins, group_names)
            df2 = one_hot_encode(df2, 'engagement')
            for k in group_names:
                cl_features.set_value(center_id, k, df2[k].mean() > 0.5)  # TODO: Pick good cut-off value

            # find count of low, medium and high reach users
            df2['influence'] = normalize(df2['influence'])
            bins = (-0.001, 0.3, 0.7, 1)  # TODO: Pick good binning values
            group_names = ('influenceLow', 'influenceMedium', 'influenceHigh')
            df2['influence'] = discretize(df2['influence'], bins, group_names)
            df2 = one_hot_encode(df2, 'influence')
            for k in group_names:
                cl_features.set_value(center_id, k, df2[k].mean() > 0.5)  # TODO: Pick good cut-off value

            # find count of low, medium and high influence users
            df2['role'] = normalize(df2['role'])
            bins = (-0.001, 0.2, 0.5, 1)  # TODO: Pick good binning values
            group_names = ('roleLow', 'roleMedium', 'roleHigh')
            df2['role'] = discretize(df2['role'], bins, group_names)
            df2 = one_hot_encode(df2, 'role')
            for k in group_names:
                cl_features.set_value(center_id, k, df2[k].mean() > 0.5)  # TODO: Pick good cut-off value

            # find the number of credible users
            temp = df2[df2['credibility'] == 1].join(df1)
            credibility = temp.drop_duplicates(subset='screen_name')['credibility']
            temp = None
            cl_features.set_value(center_id, 'credibilityMean',
                                  credibility.mean() > 0.5)  # TODO: Pick good cut-off value

            # other linguistic features
            cl_features.set_value(center_id, 'hasVulgarWords', df2['hasVulgarWords'].mean() > 0.5)
            cl_features.set_value(center_id, 'hasEmoticons', df2['hasEmoticons'].mean() > 0.5)
            cl_features.set_value(center_id, 'isInterrogative', df2['isInterrogative'].mean() > 0.5)
            cl_features.set_value(center_id, 'isExclamatory', df2['isExclamatory'].mean() > 0.5)
            cl_features.set_value(center_id, 'hasAbbreviations', df2['hasAbbreviations'].mean() > 0.5)
            cl_features.set_value(center_id, 'hasTwitterJargons', df2['hasTwitterJargons'].mean() > 0.5)
            cl_features.set_value(center_id, 'hasFPP', df2['hasFPP'].mean() > 0.5)
            cl_features.set_value(center_id, 'hasSource', df2['hasSource'].mean() > 0.5)
            cl_features.set_value(center_id, 'nr_of_sources', df2['nr_of_sources'].mean() > 0.5)
            # for key in tweet_enricher.speech_act_verbs:
            #     cl_features.set_value(index, key, df2[k].mean() > 0.5)
            cl_features.set_value(center_id, 'has#', df2['has#'].mean() > 0.5)
            cl_features.set_value(center_id, '#Position', df2['#Position'].mean() > 0.5)
            cl_features.set_value(center_id, 'hasRT', df2['hasRT'].mean() > 0.5)
            cl_features.set_value(center_id, 'RTPosition', df2['RTPosition'].mean() > 0.5)
            cl_features.set_value(center_id, 'has@', df2['has@'].mean() > 0.5)
            cl_features.set_value(center_id, '@Position', df2['@Position'].mean() > 0.5)
            cl_features.set_value(center_id, 'isRumor', 0)

            # update progress
            bar.update(i)
    else:
        raise ValueError('Feature type can only be Gaussian(G)/ Multinomial(Mn)/ Multivariate(Mv).')

    # save features
    cl_features.to_csv(os.path.join('results', os.path.splitext(save_file_name)[0] + '_cluster_features.csv'), sep='\t')


def classify(df, clf, param_grid={}):
    """
    Train and evaluates a classifier.
    :param df:
    :param clf:
    :param param_grid:
    :return: learned classifier model
    """
    assert isinstance(df, pd.DataFrame) and 'tweet_id' == df.index.name
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
