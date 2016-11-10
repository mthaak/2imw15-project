import csv
import glob
import os
from TweetEnricher.tweetEnricher import TweetEnricher

"""
    Keywords selected for initial data collection querying.
"""


data_directory = os.path.join(os.pardir, 'DataCollection', 'results')
files = []
for file in glob.glob(os.path.join(data_directory, '*_tweets.csv')):
    files.append(file)

tweet_enricher = TweetEnricher()
tweets = ''

for file in files:
    with open(os.path.splitext(file)[0] + '_unigrams.csv', 'w+', newline='', encoding='utf8') as out_file:
        writer = csv.writer(out_file)
        with open(file, 'r', encoding='utf8') as data_file:
            reader = csv.reader(data_file, delimiter='\t')
            for row in reader:
                if row[0].startswith('#') or row[0].startswith('tweet_id'):
                    continue
                tweets += row[1] + ' '
        matrix = tweet_enricher.returnUnigramMatrix([tweets])
        for w in sorted(matrix, key=matrix.get, reverse=True):
            writer.writerow([w, matrix.get(w)])
