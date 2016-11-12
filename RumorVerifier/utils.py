import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def normalize(a):
    assert isinstance(a, (pd.Series, np.ndarray, list))
    denom = float(max(a) - min(a))
    denom = 1 if denom == 0 else denom
    return (a - min(a)) / denom


def standardize(a):
    assert isinstance(a, np.ndarray)
    denom = float(max(a) - min(a))
    denom = 1 if denom == 0 else denom
    return (a - a.mean) / denom


def discretize(df, bins, group_names):
    """
    Discretize dataframe/Series/list into bins labelled by given group names
    :param df:
    :param bins:
    :param group_names:
    :return:
    """
    assert isinstance(bins, (list, tuple)) and all(isinstance(k, (int, float)) for k in bins)
    assert isinstance(group_names, (list, tuple)) and all(isinstance(k, str) for k in group_names)
    assert len(bins) == len(group_names) + 1
    return pd.cut(df, bins, labels=group_names)


def one_hot_encode(df, column_name):
    """
    Get one hot encoding of column
    :param df:
    :param column_name:
    :return:
    """
    assert isinstance(df, pd.DataFrame) and column_name in df.columns
    one_hot = pd.get_dummies(df[column_name])
    # a = df.drop(column_name, axis=1)
    for col in one_hot.columns:
        df[col] = one_hot[col]
    return df


def remove_null_values(file_name):
    assert isinstance(file_name, str)
    df = pickle.load(open(file_name, 'rb'))
    print(df.head(5))
    print(df.shape[0])
    print(df['originality'].value_counts())
    df = df[df.notnull()]
    print(df.head(5))
    print(df.shape[0])
    pickle.dump(df, open(file_name, "wb"))


if __name__ == "__main__":
    from DataCollection.utils import read_csv_ignore_comments as read_csv

    # TESTING NUMPY ARRAY TYPE CHECKING
    # l = np.array([1, 2])
    # print(type(l).__module__ == np.__name__)
    # print(isinstance(l, np.ndarray))
    # print(max(l))

    # TESTING PLOTTING FEATURE VALUES
    file_name = 'search_20161102_211623_tweets'
    # df = pickle.load(open('features_backup_(5800_processed).p', 'rb'))
    # df1 = read_csv(os.path.join(os.pardir, 'DataCollection', 'results', file_name + '.csv'), index_col='tweet_id')
    # df2 = read_csv(os.path.join('results', file_name + '_features.csv'), sep=',', index_col='tweet_id')
    # df = df2[df2['credibility'] == 1].join(df1)
    # print(df2['credibility'].value_counts())
    # print(df[['screen_name', 'credibility']].head(10))
    # print(df['credibility'].value_counts())
    # print(df.drop_duplicates(subset='screen_name')['credibility'].mean())
    # a = df['controversiality'].unique()
    # a = a[~np.isnan(a)]
    # print(a.mean())
    # b = df['controversiality']
    # # Standardize
    # b = (b - b.mean()) / b.std()
    # # Normalize
    # b = (b - b.min()) / (b.max() - b.min())
    # # b = b[b > 0.038]
    # print(b.mean(), b.std(), b.max(), b.min())
    # b.head(100).plot(kind='bar')
    # plt.show()

    # TESTING DISCRETIZE AND ONE-HOT ENCODING
    # df = df.head(1000)
    # df['controversiality'] = normalize(df['controversiality'])
    # df['originality'] = normalize(df['originality'])
    # df['controversiality'] = normalize(df['controversiality'])
    # bins = (-0.01, 0.05, 0.5, 1)
    # group_names = ('controversialityLow', 'controversialityMedium', 'controversialityHigh')
    # df['c'] = discretize(df['controversiality'], bins, group_names)
    # features = one_hot_encode(df, 'c')
    # print(features)
    # df1 = pd.DataFrame()
    # for k in group_names:
    #     df1['controversialityMean'] = df['controversiality'].apply(lambda x: features[k].mean())
    #     df1['controversialityStd'] = df['controversiality'].apply(lambda x: features[k].std())
    # print(df1['controversialityMean'])
    # print(df1['controversialityStd'])

    # FIND COLUMNS CONTAINING NaN VALUES
    df = read_csv(os.path.join('results', 'search_20161102_211623_tweets_cluster_features.csv'), index_col='center_id')
    print(pd.isnull(df).sum() > 0)
