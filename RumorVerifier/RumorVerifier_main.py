from RumorVerifier.rumorVerifier import *

if __name__ == "__main__":
    # EXTRACT FEATURES
    file_name = 'search_20161102_211623_tweets.csv'
    df = extract_features(file_name)
    # df = pickle.load(open(os.path.splitext(file_name)[0] + '_features.p', "rb"))

    # EXTRACT CLUSTER FEATURES
    # clusters = pickle.load(open(os.path.join('results', os.path.splitext(file_name)[0] + '_clusters.p')), "rb")
    # extract_cluster_features(file_name, df, clusters)

    # TRAIN CLASSIFIER
    # classify(features=df)
