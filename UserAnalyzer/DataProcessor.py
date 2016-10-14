import csv
from nltk.corpus import opinion_lexicon
from UserAnalyzer.OpinionClassifier import *


class DataProcessor:

    CL_NR_RUMOUR = 11
    CL_NR_OPINION = 12
    CL_NR_USER_ID = 1
    RATIO_OPINION = 1.5
    def __init__(self, tweets_filename, rumours_filename, users_filename):
        self.tweets_FileName = tweets_filename
        self.rumours_FileName = rumours_filename
        self.users_FileName = users_filename
        #collections of all tweets, rumours and users
        self.tweets = []
        self.rumours = []
        self.users = []
        #groups of tweets/users on the corresponding rumour
        self.tweets_rumours = dict()
        self.users_rumours = dict()
        #input for model
        self.user_features = dict()
        #label for model. Each user has one label for each rumour
        self.user_labels = dict()

    def process(self):
        self.loadData()
        self.labelOpinion()
        self.makeFeaturesOfUsers()
        self.makeLabelsOfUsers()
        self.splitTweetsAndUsers()

    def makeFeatureOfOneUser(self, id):
        """
        Function that calculates the feature of one user
        :param id:
        :return:
        """
        #TO - DO
        return [1,2,3]

    def makeLabelOfOneUser(self, userid, rumour):

        tweets_by_user = [row for row in self.tweets_rumours[rumour] if row[self.CL_NR_USER_ID] == userid]
        pos_tweets = sum(1 for t in tweets_by_user if t[DataProcessor.CL_NR_OPINION] == 1)
        neg_tweets = sum(1 for t in tweets_by_user if t[DataProcessor.CL_NR_OPINION] == -1)
        #neu_tweets = sum(1 for t in tweets_by_user if t[DataProcessor.CL_NR_OPINION] == 0)
        ratio = pos_tweets / neg_tweets
        if ratio > self.RATIO_OPINION:
            # Propagator
            list(self.user_labels[id]).append(1)
        elif ratio < 1 / self.RATIO_OPINION:
            # Stifler
            list(self.user_labels[id]).append(-1)
        else:
            # Undetermined
            list(self.user_labels[id]).append(0)
        return 0

    def loadData(self):
        """
        Function that loads the tweets and rumours. Loading data into memory makes program faster.
        :param tweets_filename:
        :param rumours_filename:
        :return:
        """
        with open(self.tweets_FileName, encoding='utf-8') as tweetsFile:
            tweets = csv.reader(tweetsFile, delimiter='\t')
            for row in tweets:
                self.tweets.append(row)

        with open(self.rumours_FileName, encoding='utf-8') as rumoursFile:
            rumours = csv.reader(rumoursFile, delimiter='\t')
            for row in rumours:
                self.rumours.append(row)

        with open(self.users_FileName, encoding='utf-8') as usersFile:
            users = csv.reader(usersFile, delimiter='\t')
            for row in users:
                self.users.append(row)

    def labelOpinion(self, opinionClassifier):
        """
        Function that labels the opinion of all tweets about a rumour
        :param opinionClassifier
        :return: none.
        """
        for t in self.tweets:
            t[DataProcessor.CL_NR_OPINION] = opinionClassifier.predictOpinion()
        return

    def makeFeaturesOfUsers(self):
        """
        Function that calculates the feature of all users
        :return:
        """
        # naive features
        for row in self.users:
            id = row[DataProcessor.CL_NR_USER_ID]
            feature = self.makeFeatureOfOneUser(id)
            self.user_features[id].append(feature)

    def makeLabelsOfUsers(self):
        """
        Function that calculates the label of all users (Propagator, Stifler, undetermined)
        :return:
        """
        for rumour, users in self.users_rumours.items():
            for user in users:
                # make label of a user
                self.user_labels[user][rumour] = self.makeLabelOfOneUser(user, rumour)
        return

    def splitTweetsAndUsers(self):
        """
        Function that splits tweets and users into groups of rumours.
        :return:
        """
        for row in self.tweets:
            rumour = row[DataProcessor.CL_NR_RUMOUR]
            self.tweets_rumours[rumour].append(row)
            self.users_rumours[rumour].append(row[DataProcessor.CL_NR_USER_ID])

    def outputData(self, rumour):
        users = self.users_rumours[rumour]
        X = self.user_features[users]
        y = []
        for labels in self.user_labels[users]:
            y.append(labels[rumour])
        return X, y

