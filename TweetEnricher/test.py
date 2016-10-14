import csv
from TweetEnricher.tweetEnricher import TweetEnricher

INPUT_FILE = "../DataCollection/results/vote_leave_tweets.csv"
OUTPUT_FILE = "../DataCollection/results/vote_leave_tweets_unigrams.csv"

tweet_enricher = TweetEnricher()
tweets=''

with open(OUTPUT_FILE,'w+', newline='',encoding='utf8') as out_file:
    writer = csv.writer(out_file)
    with open(INPUT_FILE,'r',encoding='utf8') as data_file:
        reader = csv.reader(data_file, delimiter='\t')
        data_file.readline()
        for i, row in enumerate(reader):
            tweets+=row[1]
    matrix = tweet_enricher.returnUnigramMatrix([tweets])
    for w in matrix:
        writer.writerow([w, matrix.get(w)])

