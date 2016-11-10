from RumorVerifier.rumorVerifier import *
from DataCollection.utils import read_csv_ignore_comments as read_csv
import csv
from sklearn.naive_bayes import GaussianNB
import ast


def parse_bad_clusters(path):
    c = []
    with open(path, mode='r', encoding='utf8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            row = ''.join(row).replace('\t', '')
            row = row.replace('][', '],[')
            row = row.replace("' '", "','")
            c.append(ast.literal_eval(row))
    print(c)
    return c

if __name__ == "__main__":
    # read tweets data
    file_path = os.path.join(os.pardir, 'DataCollection', 'results', 'search_20161102_211623_tweets.csv')
    df1 = read_csv(file_path, index_col='tweet_id')

    # extract features
    # df2 = extract_features(df1, coe_backup_file='features_backup_(6000_processed).p')
    # df2 = pickle.load(open(os.path.splitext(file_name)[0] + '_features.p', "rb"))

    # extract cluster features
    clusters = parse_bad_clusters(path=os.path.join(os.pardir, 'Data',
                                                    'search_20161102_211623_tweets_clustered_(1000).csv'))

    # clusters = pickle.load(open(os.path.join('results', os.path.splitext(file_name)[0] + '_clusters.p')), "rb")
    extract_cluster_features(df1, df2, feature_type='G', clusters=clusters)

    # classify the clusters
    classify(df=df2, clf=GaussianNB, param_grid={})
