import pickle
import csv

from UserAnalyzer.OpinionClassifier import *
from UserAnalyzer.DataProcessor import *
from UserAnalyzer.MarkovChain import *


friends_map = {
    "a" : {"b" },
    "b" : {"c"},
    "c" : {"d"},
    "d" : {"a"}
};

users = [
    {"id":"a", "retweets": 80, "likes": 10, "ei": 0, "teleportation": 0},
    {"id":"b", "retweets": 20, "likes": 10, "ei": 0, "teleportation": 0},
    {"id":"c", "retweets": 100, "likes": 10, "ei": 0, "teleportation": 0},
    {"id":"d", "retweets": 40, "likes": 10, "ei": 0, "teleportation": 0}
]


testOC = OpinionClassifier()
testDP = DataProcessor("tweets_brexit.csv", "tweets_brexit.csv", "tweets_brexit.csv")
testDP.loadData();
testMC = MarkovChain(users, friends_map)
#testMC.load_data("search_20161102_211623_tweets.csv", "user_map.csv")
testMC.calc_influence()
#for t in testDP.tweets[:10]:
#    print(t[1])
#    o = testOC.predict_opinion(t[1])

