import pandas as pd
import os


def merge_csvs(files):
    """ Merge given CSV files. Input files must contain the same header. """
    assert isinstance(files, list) and all(hasattr(f, 'read') for f in files)
    df = pd.concat((pd.read_csv(f, sep='\t', index_col=None, header=0) for f in files))
    df.to_csv(os.path.join('results', 'out.csv'), index=False, sep='\t', encoding='utf-8')

if __name__ == "__main__":
    users = ['vote_leave', 'BorisJohnson', 'David_Cameron',
             'Nigel_Farage', 'michaelgove', 'George_Osborne']
    list_of_files = [
        open(os.path.join('results', '%s_tweets.csv' % user), 'r', encoding='utf-8') for user in users]
    merge_csvs(list_of_files)
