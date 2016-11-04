import pandas as pd
import os
import ast
import csv
from calendar import monthrange
from datetime import timedelta
import nltk
import re
import string


def read_csv_ignore_comments(file_path, sep="\t", index_col=None, comment_literal='#'):
    with open(file_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        skip = [i for i, row in enumerate(r) if row[0].startswith(comment_literal)]
        f.seek(0)
        df = pd.read_csv(f, sep=sep, index_col=index_col, header=0, skiprows=skip, skip_blank_lines=True)
    return df


def merge_csvs(files):
    """ Merge given CSV files. Input files must contain the same header. """
    assert isinstance(files, list) and all(hasattr(f, 'read') for f in files)
    df = pd.concat((pd.read_csv(f, sep='\t', index_col=None, header=0) for f in files))
    df.to_csv(os.path.join('results', 'out.csv'), index=False, sep='\t', encoding='utf-8')


def days_delta(from_date, to_date):
    """
    Calculate the number of months from from_date to to_date.
    :param from_date:
    :param to_date:
    :return:
    """
    delta = 0
    while True:
        mdays = monthrange(from_date.year, from_date.month)[1]
        from_date += timedelta(days=mdays)
        if from_date <= to_date:
            delta += mdays
        else:
            break
    return delta


def month_delta(from_date, to_date):
    """
    Calculate the number of months from from_date to to_date.
    :param from_date:
    :param to_date:
    :return:
    """
    delta = 0
    while True:
        mdays = monthrange(from_date.year, from_date.month)[1]
        from_date += timedelta(days=mdays)
        if from_date <= to_date:
            delta += 1
        else:
            break
    return delta


def filter_by_keywords(df, list_of_keywords):
    """
    Filters the tweets (df) based on AND
    combinations of inputted keywords
    """
    assert isinstance(list_of_keywords, list) and all(isinstance(k, str) for k in list_of_keywords)
    return df[df.keywords.apply(ast.literal_eval).apply(lambda x: any(k in x for k in list_of_keywords))]


def print_progress_bar(index, length):
    # from time import sleep
    import sys, math
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('=' * math.ceil(20 * index / length), 100 * index / length))
    sys.stdout.flush()
    # sleep(0.25)


ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}


def clean_text(text, log=False):
    assert isinstance(text, str)
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    if log: print(text)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    if log: print(text)
    text = re.sub("@\w*", "", text)
    if log: print(text)
    text = re.sub("#\w*", "", text)
    if log: print(text)
    text = re.sub('[^\x00-\x7F]', "", text)
    if log: print(text)
    return text


def get_language(text):
    assert isinstance(text, str)
    text = clean_text(text)
    words = set(nltk.wordpunct_tokenize(text))
    words = set(w for w in words if len(w) > 1 and w not in list(string.punctuation)
                + ["…", "...", "..", ")", "(", "-->", "->", ">>", "#", "rt", "@"])
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key=lambda x: x[1])[0]


def is_english(text, log=False):
    assert isinstance(text, str)
    text = clean_text(text, log)
    words = set(nltk.wordpunct_tokenize(text))
    # words = set(text.split())
    words = set(w for w in words if len(w) > 1 and w not in list(string.punctuation)
                + ["…", "...", "..", ")", "(", "-->", "->", ">>", "#", "rt", "@"])
    if log: print(words)
    return len(words & ENGLISH_STOPWORDS) >= len(words & NON_ENGLISH_STOPWORDS)


def remove_duplicated_spaces(text):
    assert isinstance(text, str)
    return " ".join(text.split())


if __name__ == "__main__":
    # users = ['vote_leave', 'BorisJohnson', 'David_Cameron',
    #          'Nigel_Farage', 'michaelgove', 'George_Osborne']
    # list_of_files = [
    #     open(os.path.join('results', '%s_tweets.csv' % user), 'r', encoding='utf-8') for user in users]
    # # merge_csvs(list_of_files)
    #
    # # Test to parse list from csv fields
    # df = read_csv_ignore_comments(list_of_files[0], index_col='tweet_id')
    # print(ast.literal_eval(df['urls'][0])[0])
    #
    # # Test keywords filter
    # print(filter_by_keywords(df, ['people']))

    df = read_csv_ignore_comments(os.path.join('results', 'search_20161102_211623_tweets.csv'), index_col='tweet_id')

    # Fixing missing reply_to_id's
    # df['is_reply'] = df['is_reply'].apply(lambda x: 1 if x else 0)
    # df['reply_to_id'] = df.apply(lambda x: -1 if x['is_reply'] == 0 else x['reply_to_id'], axis=1).astype('int64')
    # df.to_csv(os.path.join('results', 'search_20161102_211623_tweets.csv'), sep='\t', encoding='utf-8')

    # Filtering out non-english tweets
    # df['is_english'] = df['text'].apply(is_english)
    # print(df.loc[df['is_english']==True, 'text'].head(10))
    # print('======================================')
    # print(df.loc[df['is_english']==False, 'text'])
    # print('======================================')
    # print(df['is_english'].value_counts())
    # print('======================================')
    # is_english("See also Brexit. https://t.co/o99kG0hDfg", True)
    # print('======================================')

    # Fixing people without screen_names
    # print(df['screen_name'].apply(lambda x: len(str(x)) == 0).value_counts())
