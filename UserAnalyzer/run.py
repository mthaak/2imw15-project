import numpy as np
from numpy import linalg as LA

friends_map = {
    "a" : {"b" },
    "b" : {"c"},
    "c" : {"d"},
    "d" : {"a"}
};

users = [
    {"id":"a", "retweets": 0, "likes": 10, "ei": 0, "teleportation": 0},
    {"id":"b", "retweets": 10, "likes": 10, "ei": 0, "teleportation": 0},
    {"id":"c", "retweets": 20, "likes": 10, "ei": 0, "teleportation": 0},
    {"id":"d", "retweets": 30, "likes": 10, "ei": 0, "teleportation": 0}
]























############
nr_users = len(users)

P = [[0 for x in range(nr_users)] for y in range(nr_users)]
a = 0.4
b = 1-a

ei_total = 0

for u in users:
    u['ei'] = u['retweets'] + u['likes']
    ei_total += u['ei']

for u in users:
    u['teleportation'] = a*u['ei']/ei_total

row_ei_sum = np.full(nr_users, 0)

for i in range(nr_users):
    for j in range(nr_users):
        if users[j]['id'] in friends_map[users[i]['id']]:
            P[i][j] = users[j]['ei']
            row_ei_sum[i] += users[j]['ei']

for i in range(nr_users):
    if not friends_map[users[i]['id']]:
        for j in range(nr_users):
            P[i][j] = users[j]['teleportation']
    else:
        for j in range(nr_users):
            P[i][j] = b*P[i][j]/row_ei_sum[i] + users[j]['teleportation']



prev_influences = np.full(nr_users,1/nr_users)
new_influence = np.full(nr_users, 1/nr_users)
diff = np.full(nr_users,  1/nr_users)

power = 1
max_power = 2^14

while( np.sum(diff) > (0.001*nr_users) and power <= max_power):
    Pk = LA.matrix_power(P, power)
    new_influence = np.dot(prev_influences, Pk)
    diff = new_influence - prev_influences
    prev_influences = new_influence
    power = power *2
print(new_influence)
