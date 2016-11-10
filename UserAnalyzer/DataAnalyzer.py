import csv
import pickle

class DataAnalyzer:

    def __init__(self, tweets_filename, rumours_filename, users_filename):
        self.USER_CL_FOLLOWERS = 1
        self.USER_CL_FOLLOWINGS = 2
        self.USER_CL_TWEETS = 3
        self.USER_CL_RETWEETS = 4
        self.USER_CL_ID = 0

        self.TWEET_CL_RETWEET = 1
        self.TWEET_CL_USER_ID = 0

        self.RUMOUR_CL_RETWEET = 3
        self.RUMOUR_CL_OPINION=4

        #files
        self.tweets_FileName = tweets_filename
        self.rumours_FileName = rumours_filename
        self.users_FileName = users_filename
        self.users = []
        self.tweets = []
        self.rumours = []

        self.results = []
    def loadData(self):
        """
        Function that loads the tweets and rumours. Loading data into memory makes program faster.
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
                self.results[row[self.USER_CL_ID]] = {
                                     "alpha": 0,
                                     "influence": 0,
                                     "credibility": 0,
                                     "rumours_produced": 0,
                                     "rumours_propagated": 0,
                                     "rumours_denied": 0}


    def calc_influence(self):
        for row in self.users:
            user_id = row[self.USER_CL_ID]
            followers = row[self.USER_CL_FOLLOWERS]
            followings = row[self.USER_CL_FOLLOWINGS]
            tweets = row[self.USER_CL_TWEETS]
            retweets = row[self.USER_CL_RETWEETS]

            #alpha
            alpha = followers/followings+tweets/retweets
            self.results[user_id]["alpha"] = alpha

            #influence calculation
            a = followers/(followings+followers)
            b = retweets/(retweets+tweets)
            influence = 2*a*b/(a+b);
            self.results[user_id]["influence"]=influence

    def calc_rumours(self):
        for row in self.rumours:
            opinion = row[self.RUMOUR_CL_OPINION]
            if opinion == 1:
                if row[self.RUMOUR_CL_RETWEET] == 0:
                    self.results[row[self.TWEET_CL_USER_ID]]["rumours_produced"] += 1
                else:
                    self.results[row[self.TWEET_CL_USER_ID]]["rumours_propagated"] += 1
            elif opinion == -1:
                self.results[row][row[self.TWEET_CL_USER_ID]]["rumours_denied"] += 1

    def output(self):
        with open('results.obj', 'wb') as fp:
            pickle.dump(self.results, fp)

    def solve_credibility(self, graphs):
        return 0
