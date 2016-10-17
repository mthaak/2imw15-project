import csv
from TweetEnricher.tweetEnricher import TweetEnricher
import glob, os

INPUT_FILE =[]
DIRECTORY = "../DataCollection/results/"

os.chdir(DIRECTORY)
for file in glob.glob("*_tweets.csv"):
    INPUT_FILE.append(file)
os.chdir("../../TweetEnricher/")

tweet_enricher = TweetEnricher()
tweets=''

for file in INPUT_FILE:
    with open(DIRECTORY+"Unigrams_"+file,'w+', newline='',encoding='utf8') as out_file:
        writer = csv.writer(out_file)
        with open(DIRECTORY+file,'r',encoding='utf8') as data_file:
            reader = csv.reader(data_file, delimiter='\t')
            data_file.readline()
            for i, row in enumerate(reader):
                tweets+=row[1]
        matrix = tweet_enricher.returnUnigramMatrix([tweets])
        for w in sorted(matrix, key=matrix.get, reverse=True):
            writer.writerow([w, matrix.get(w)])

