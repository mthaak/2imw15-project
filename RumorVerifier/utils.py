import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def normalize(a):
    assert isinstance(a, (np.ndarray, list))
    return (a - min(a)) / float(max(a) - min(a))


def standardize(a):
    assert isinstance(a, np.ndarray)
    return (a - a.mean) / float(max(a) - min(a))


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
    assert len(group_names) == len(bins) + 1
    return pd.cut(df, bins, labels=group_names)


def one_hot_encode(df, column_name):
    """
    Get one hot encoding of column
    :param df:
    :param column_name:
    :return:
    """
    one_hot = pd.get_dummies(df[column_name])
    # a = df.drop(column_name, axis=1)
    return df.join(one_hot)


if __name__ == "__main__":
    # l = np.array([1, 2])
    # print(type(l).__module__ == np.__name__)
    # print(isinstance(l, np.ndarray))
    # print(max(l))

    from DataCollection.utils import read_csv_ignore_comments as read_csv

    file_name = 'search_20161102_211623_tweets'
    # df = pickle.load(open('features_backup_(3500_processed).p', 'rb'))
    df1 = read_csv(os.path.join(os.pardir, 'DataCollection', 'results', file_name + '.csv'), index_col='tweet_id')
    df2 = read_csv(os.path.join('results', file_name + '_features.csv'), sep=',', index_col='tweet_id')
    df = df2[df2['credibility'] == 1].join(df1)
    print(df2['credibility'].value_counts())
    print(df[['screen_name', 'credibility']].head(10))
    print(df['credibility'].value_counts())
    print(df.drop_duplicates(subset='screen_name')['credibility'].mean())
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
