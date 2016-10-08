import tweepy
import time
import csv
import pickle

################################################
# INSTANTIATE API
################################################


def switch_auth(idx):
    """ Switch current api authentication """
    if idx >= len(api.auths):
        raise IndexError('Index out of bounds.')
    api.auth_idx = idx
    api.auth = api.auths[idx]


def handle_rate_limit(resource, path):
    """ Switch authentication from the current one which is depleted """
    _rate_limit_status = []
    for auth in api.auths:
        api.auth = auth
        result = api.rate_limit_status()['resources'][resource][path]
        _rate_limit_status.append(result)
    idx = max(enumerate(_rate_limit_status), key=lambda x: x[1]['remaining'])[0]
    if _rate_limit_status[idx]['remaining'] > 0:
        switch_auth(idx)
    else:
        next_idx = min(enumerate(_rate_limit_status), key=lambda x: x[1]['reset'])[0]
        sleep_time = _rate_limit_status[next_idx]['reset'] - int(time.time())
        if sleep_time > 0:
            time.sleep(sleep_time + 5)
        switch_auth(next_idx)


def remaining_calls(resource, path):
    """ Get the remaining number of calls left for a given API resource """
    result = api.rate_limit_status()['resources'][resource][path]['remaining']
    print(result)
    return result

# Load all the OAuth handlers
auths = []
credentialsFile = open('twitter_credentials.csv', 'r')
credentialsReader = csv.DictReader(credentialsFile)
for cred in credentialsReader:
    print(cred)
    auth = tweepy.OAuthHandler(cred['consumer_key'], cred['consumer_secret'])
    auth.set_access_token(cred['access_token'], cred['access_secret'])
    auths.append(auth)
if not auths:
    raise ValueError('No OAuth handlers available.')

# Load the Twitter API
api = tweepy.API(auths[0], retry_count=3, retry_delay=5,
                 retry_errors={401, 404, 500, 503})
api.auths = list(auths)
api.auth_idx = 0
auths = None

################################################
# COLLECT TWEETS
################################################

# Open a CSV file to store the results
csvFile = open('result.csv', 'a')
csvWriter = csv.writer(csvFile, delimiter='\t')

# List of users
users = pickle.load(open("users.p", "rb"))

# Resource from which we want to collect tweets
resource, path = 'statuses', '/statuses/user_timeline'


def list_tweets(cursor):
    """ Iterate over the tweets with a cursor """
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            handle_rate_limit(resource, path)

# Get tweets from given user
for user in users:
    for tweet in list_tweets(tweepy.Cursor(api.user_timeline, id=user, count=1).items()):
        print(tweet._json, '\n')
        csvWriter.writerow([tweet.author.id,
                            tweet.author.name,
                            tweet.id,
                            tweet.created_at,
                            tweet.text.encode('utf-8'),
                            tweet.retweet_count,
                            tweet.author.followers_count,
                            tweet.author.friends_count,
                            tweet.author.statuses_count])
