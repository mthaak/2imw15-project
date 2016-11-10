from RumorVerifier.rumorVerifier import *
from DataCollection.utils import read_csv_ignore_comments as read_csv
import csv
from sklearn.naive_bayes import GaussianNB
import ast

if __name__ == "__main__":
    # read tweets data
    file_name = 'search_20161102_211623_tweets.csv'
    # df = read_csv(os.path.join(os.pardir, 'DataCollection', 'results', file_name), index_col='tweet_id')

    # extract features
    df = extract_features(file_name, coe_backup_file='features_backup_(6000_processed).p')
    # df = pickle.load(open(os.path.splitext(file_name)[0] + '_features.p', "rb"))

    # extract cluster features
    clusters = []
    with open(os.path.join(os.pardir, 'Data', 'tweets_20161024_111847_clustered(1000).csv'),
              mode='r', encoding='utf8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            row = ''.join(row).replace('\t', '')
            row = row.replace('][', '],[')
            row = row.replace("' '", "','")
            clusters.append(ast.literal_eval(row))
    print(clusters)
    # clusters = pickle.load(open(os.path.join('results', os.path.splitext(file_name)[0] + '_clusters.p')), "rb")
    extract_cluster_features(file_name, df, feature_type='G', clusters=clusters)

    # classify the clusters
    classify(df=df, clf=GaussianNB, param_grid={})
