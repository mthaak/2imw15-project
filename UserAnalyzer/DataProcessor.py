import csv
from nltk.corpus import opinion_lexicon
from UserAnalyzer.OpinionClassifier import *
from UserAnalyzer.UserClassifier import *
class DataProcessor:

    CL_NR_RUMOUR = 11
    CL_NR_OPINION = 12
    CL_NR_USER_ID = 1
    CL_NR_TWEET = 1

    CL_NR_FOLLOWERS = 2
    CL_NR_FOLLOWINGS = 3
    RATIO_OPINION = 1.5

    def __init__(self, tweets_filename, rumours_filename, users_filename):
        #files
        self.tweets_FileName = tweets_filename
        self.rumours_FileName = rumours_filename
        self.users_FileName = users_filename
        #models
        self.models = dict()
        #collections of all tweets, rumours and users
        self.tweets = []
        self.rumours = []
        self.users = []
        #groups of tweets/users on the corresponding rumour.
        self.tweets_rumours = dict() #key: rumour
        self.users_rumours = dict() #key: rumour
        #input for model.
        self.user_features = dict() #key: user_id
        #label for model. Each user has one label for each rumour
        self.user_labels = dict() #key: user_id

    def process(self, models):
        self.loadData()
        self.loadModels(models)
        self.initTweetsProcess()
        self.makeFeaturesOfUsers()
        self.makeLabelsOfUsers()

    def makeFeatureOfOneUser(self, id):
        """
        Function that calculates the feature of one user
        :param id:
        :return: Array of feature
        """
        #to-do
        #1. #follower/#followings
        user = self.users[id]
        feature1 = user[DataProcessor.CL_NR_FOLLOWERS]/user[DataProcessor.CL_NR_FOLLOWINGS]
        #2.
        return [1, 2, 3]

    def makeLabelOfOneUser(self, userid, rumour):
        """
        Function that determines the category of user. (propagator, stifler, undertermined)
        :param userid:
        :param rumour:
        :return:
        """
        tweets_by_user = [row for row in self.tweets_rumours[rumour] if row[self.CL_NR_USER_ID] == userid]
        pos_tweets = sum(1 for t in tweets_by_user if t[DataProcessor.CL_NR_OPINION] == 1)
        neg_tweets = sum(1 for t in tweets_by_user if t[DataProcessor.CL_NR_OPINION] == -1)
        # neu_tweets = sum(1 for t in tweets_by_user if t[DataProcessor.CL_NR_OPINION] == 0)
        ratio = pos_tweets / neg_tweets
        if ratio > self.RATIO_OPINION:
            # Propagator
            self.user_labels[id][rumour] = 1
        elif ratio < 1 / self.RATIO_OPINION:
            # Stifler
            self.user_labels[id][rumour] = -1
        else:
            # Undetermined
            self.user_labels[id][rumour] = 0

    def loadData(self):
        """
        Function that loads the tweets and rumours. Loading data into memory makes program faster.
        :return:
        """
        with open(self.tweets_FileName, encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            next(reader)  # skip header
            for i, row in enumerate(reader):
                self.tweets.append(row)

        with open(self.rumours_FileName, encoding='utf-8') as rumoursFile:
            reader = csv.reader(rumoursFile, delimiter='\t')
            next(reader)  # skip header
            for row in enumerate(reader):
                self.rumours.append(row)

        with open(self.users_FileName, encoding='utf-8') as usersFile:
            reader = csv.reader(usersFile, delimiter='\t')
            for row in enumerate(reader):
                self.users.append(row)

    def loadModels(self, models):
        """
        Function that loads all necessary models. Currently only OpinionClassifier is needed.
        :param models: Array of Model
        :return:
        """
        for name, model in models:
            self.models[name] = model

    def initTweetsProcess(self):
        """
        Function that splits tweets, users into groups of rumours and label opinions.
        :return:
        """
        for row in self.tweets:
            rumour = row[DataProcessor.CL_NR_RUMOUR]
            row[DataProcessor.CL_NR_OPINION] = \
                self.models["OpinionClassifier"].predictOpinion(row[self.CL_NR_TWEET], self.rumours[rumour])
            self.tweets_rumours[rumour].append(row)
            self.users_rumours[rumour].append(row[DataProcessor.CL_NR_USER_ID])

    def makeFeaturesOfUsers(self):
        """
        Function that calculates the feature of all users
        :return:
        """
        for row in self.users:
            id = row[DataProcessor.CL_NR_USER_ID]
            feature = self.makeFeatureOfOneUser(id)
            self.user_features[id] = feature

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

    def outputData(self, rumour):
        users = self.users_rumours[rumour]
        X = self.user_features[users]
        y = []
        for labels in self.user_labels[users]:
            y.append(labels[rumour])
        return X, y

