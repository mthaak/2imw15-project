import pandas as pd
import os
import ast


def merge_csvs(files):
    """ Merge given CSV files. Input files must contain the same header. """
    assert isinstance(files, list) and all(hasattr(f, 'read') for f in files)
    df = pd.concat((pd.read_csv(f, sep='\t', index_col=None, header=0) for f in files))
    df.to_csv(os.path.join('results', 'out.csv'), index=False, sep='\t', encoding='utf-8')


def filter_by_keywords(df, list_of_keywords):
    """
    Filters the tweets (df) based on AND
    combinations of inputted keywords
    """
    assert isinstance(list_of_keywords, list) and all(isinstance(k, str) for k in list_of_keywords)
    return df[df.keywords.apply(ast.literal_eval).apply(lambda x: all(k in x for k in list_of_keywords))]

if __name__ == "__main__":
    users = ['vote_leave', 'BorisJohnson', 'David_Cameron',
             'Nigel_Farage', 'michaelgove', 'George_Osborne']
    list_of_files = [
        open(os.path.join('results', '%s_tweets.csv' % user), 'r', encoding='utf-8') for user in users]
    # merge_csvs(list_of_files)

    # Test to parse list from csv fields
    df = pd.read_csv(list_of_files[0], sep="\t", index_col=None, header=0)
    print(ast.literal_eval(df['urls'][0])[0])

    # Test keywords filter
    print(filter_by_keywords(df, ['people']))
