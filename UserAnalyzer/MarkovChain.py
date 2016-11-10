import numpy as np
from numpy import linalg as LA
import csv


class MarkovChain:

    def __init__(self, user_profile, friends_map):
        self.friends_map = friends_map
        self.users = user_profile
        #self.friends_map = {}

        #self.users_raw = []
        #self.users_screenname = []
        #self.load_data( tweets_filename, user_map_filename)
        self.nr_users = len(self.users)
        self.P = [[0 for x in range(self.nr_users)] for y in range(self.nr_users)]
        self.a = 0.4
        self.b = 1 - self.a
        self.ei_total = 0
        return

    def load_data(self, tweets_filename, user_map_filename):
        limit = 1000
        counter = 0
        with open(tweets_filename, encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            next(reader)  # skip header
            for i, row in enumerate(reader):
                if counter == 1000:
                    break
                counter += 1
                self.tweets.append(row)
                self.users_raw.append(row[0])
                #print(row[3])
                self.users_screenname.append(row[7])
                newuser =  {"id": row[0], "retweets": int(row[3]), "likes": 0, "ei": 0, "teleportation": 0}
                print(newuser)
                self.users.append(newuser)
        with open(user_map_filename, encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            #next(reader)  # skip header
            for i, row in enumerate(reader):
                if i == 1000:
                    break;
                self.friends_map[self.users_raw[i]] = row[1]
            #print(self.friends_map)
        print(len(self.users_raw))
        print(len(self.users))
        print(len(self.friends_map))
    def clean(self):
        self.nr_users = len(self.users)
        self.P = [[0 for x in range(self.nr_users)] for y in range(self.nr_users)]
        self.a = 0.4
        self.b = 1 - self.a
        self.ei_total = 0
    def calc_influence(self):
        for u in self.users:
            u['ei'] = u['retweets'] + u['likes']
            self.ei_total += u['ei']

        for u in self.users:
            u['teleportation'] = self.a * u['ei'] / self.ei_total

        row_ei_sum = np.full(self.nr_users, 0)

        for i in range(self.nr_users):
            for j in range(self.nr_users):
                if self.users[j]['id'] in self.friends_map[self.users[i]['id']]:
                    self.P[i][j] = self.users[j]['ei']
                    row_ei_sum[i] += self.users[j]['ei']

        for i in range(self.nr_users):
            if not self.friends_map[self.users[i]['id']]:
                for j in range(self.nr_users):
                    self.P[i][j] = self.users[j]['teleportation']
            else:
                for j in range(self.nr_users):

                    self.P[i][j] = self.b * self.P[i][j] / (row_ei_sum[i]) + self.users[j]['teleportation']
        prev_influences = np.full(self.nr_users, 1 / self.nr_users)
        diff = np.full(self.nr_users, 1 / self.nr_users)

        power = 1
        max_power = 2 ^ 14

        while (np.sum(diff) > 0.001 * self.nr_users and power <= max_power):
            Pk = LA.matrix_power(self.P, power)
            new_influence = np.dot(prev_influences, Pk)
            diff = new_influence - prev_influences
            prev_influences = new_influence
            power = power * 2
        print(new_influence)
