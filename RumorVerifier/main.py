from RumorVerifier.rumorVerifier import *
from DataCollection.utils import read_csv_ignore_comments as read_csv

if __name__ == "__main__":
    # read tweets data
    file_name = 'search_20161102_211623_tweets.csv'
    df = read_csv(os.path.join(os.pardir, 'DataCollection', 'results', file_name))

    # EXTRACT FEATURES
    df = extract_features(file_name, coe_backup_file='features_backup_(4800_processed).p')
    # df = pickle.load(open(os.path.splitext(file_name)[0] + '_features.p', "rb"))

    # EXTRACT CLUSTER FEATURES
    # clusters = pickle.load(open(os.path.join('results', os.path.splitext(file_name)[0] + '_clusters.p')), "rb")
    # extract_cluster_features(file_name, df, clusters)

    # Classify the clusters
    classify(df=df, clf=GaussianNB, param_grid={})
