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
    c = [(list(map(int, k[0])), int(k[1][0])) for k in c]
    print(c)
    with open(os.path.join('results', os.path.basename(path)),
              mode='w', newline='') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(['tweet_ids', 'center_id'])
        writer.writerows(c)
    return c

if __name__ == "__main__":
    # read tweets data
    file_name = 'search_20161102_211623_tweets'
    file_path = os.path.join(os.pardir, 'DataCollection', 'results', file_name + '.csv')
    df1 = read_csv(file_path, index_col='tweet_id')

    # extract features
    # coe_backup_file = pickle.load(open('features_backup_(6000_processed).p', 'rb'))
    # df2 = extract_features(df1, features=coe_backup_file, save_file_name=file_name)
    df2 = pickle.load(open(os.path.join('results', file_name + '_features.p'), 'rb'))

    # extract cluster features
    # clusters = parse_bad_clusters(path=os.path.join(os.pardir, 'Data',
    #                                                 'search_20161102_211623_tweets_clusters_(1000).csv'))
    # clusters = read_csv(os.path.join('results', 'search_20161102_211623_tweets_clusters_(1000).csv'))
    # clusters = clusters.set_index('center_id')
    # clusters['tweet_ids'] = clusters['tweet_ids'].apply(ast.literal_eval)
    # df2 = extract_cluster_features(df1, df2, feature_type='G', clusters=clusters, save_file_name=file_name)
    df2 = read_csv(os.path.join('results', 'search_20161102_211623_tweets_cluster_features.csv'), index_col='center_id')

    # classify the clusters
    classify(features=df2, clf=GaussianNB(), param_grid={})
