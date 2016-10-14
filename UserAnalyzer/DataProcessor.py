import csv
from nltk.corpus import opinion_lexicon


class DataProcessor:
    CL_NR_RUMOUR_LABEL = 10
    CL_NR_RUMOUR_TOPIC = 11
    CL_NR_OPINION_LABEL = 12
    CL_NR_USER_ID = 1
    RATIO_OPINION = 1.4
    test = 0

    def __init__(self, tweets_filename):
        self.rumour_tweets = []
        self.features = []
        self.labels = []
        self.Tweets_FileName =tweets_filename
        self.user_topics_groups = {}
        self.tweet_topics_groups = {}
        #independent of rumours
        self.user_features = {}
        #dependent on rumours
        self.user_labels = {}

    def process(self):
        self.splitRumours();
        self.labelOpinion();
        self.makeFeatures();
        #self.makeUserFeatures();
        #self.makeUserLabels();

    def splitRumours(self):
        with open(self.Tweets_FileName, encoding='utf-8') as tweetsFile:
            tweets = csv.reader(tweetsFile, delimiter='\t')
            for row in tweets:
                if row[DataProcessor.CL_NR_RUMOUR_LABEL] == 'R':
                    self.user_features[row[DataProcessor.CL_NR_USER_ID]] = []
                    self.rumour_tweets.append(row)
                    topic = row[DataProcessor.CL_NR_RUMOUR_TOPIC]
                    list(self.tweet_topics_groups[topic]).append(row)
                    list(self.user_topics_groups[topic]).append(row[DataProcessor.CL_NR_USER_ID])

    def labelOpinion(self):
        for tweet in self.rumour_tweets:
            if tweet == "ye":
                self.labels.append(1)
                #tweet[self.CL_NR_OPINION_LABEL] = 1
            elif tweet == "nope":
                self.labels.append(-1)
                #tweet[self.CL_NR_OPINION_LABEL] = -1
            else:
                self.labels.append(0)
                #tweet[self.CL_NR_OPINION_LABEL] = 0

    def makeFeatures(self):
        # naive features
        for row in self.rumour_tweets:
            feature = [row[6], row[7], row[8]]
            self.features.append(feature)

    def makeUserFeatures(self):
        for user_id in self.user_features.keys():
            #naive features
            basicInfo = next(row for row in self.rumour_tweets if row[self.CL_NR_USER_ID] == user_id)
            self.user_features[user_id] = [basicInfo[6], basicInfo[7], basicInfo[8]]

    def makeUserLabels(self):
        for topic, group in self.user_topics_groups.items():
            for user in group:
                # make label of a user
                tweets_by_user = [row for row in self.tweet_topics_groups[topic] if row[self.CL_NR_USER_ID] == user]
                pos_tweets = sum(1 for t in tweets_by_user if t[DataProcessor.CL_NR_OPINION_LABEL] == 1)
                neg_tweets = sum(1 for t in tweets_by_user if t[DataProcessor.CL_NR_OPINION_LABEL] == -1)
                ratio = pos_tweets/neg_tweets
                if ratio > self.RATIO_OPINION:
                    #Propagator
                    list(self.user_labels[topic]).append(1)
                elif ratio < 1/self.RATIO_OPINION:
                    #Stifler
                    list(self.user_labels[topic]).append(-1)
                else:
                    #Undetermined
                    list(self.user_labels[topic]).append(0)
        return

    def outputData(self, topic):
        X = self.user_features[ self.user_topics_groups[topic]]
        y = self.user_labels[topic]
        return X, y

    def getTopics(self):
        return self.user_topics_groups.keys()