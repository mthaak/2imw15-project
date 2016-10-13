import tweepy
import time
import csv
import pickle

################################################
# INSTANTIATE API
################################################


def switch_auth(idx):
    """ Switch current api authentication """
    assert isinstance(idx, int)
    if idx >= len(api.auths):
        raise IndexError('Index out of bounds.')
    api.auth_idx = idx
    api.auth = api.auths[idx]


def handle_rate_limit(resource, path):
    """ Switch authentication from the current one which is depleted """
    assert isinstance(resource, str) and isinstance(path, str)

    # Get rate limit status of all OAuth credentials
    _rate_limit_status = []
    for auth in api.auths:
        api.auth = auth
        result = api.rate_limit_status()['resources'][resource][path]
        _rate_limit_status.append(result)

    # Check if maximum remaining calls in all auths is > 0
    idx = max(enumerate(_rate_limit_status), key=lambda x: x[1]['remaining'])[0]
    if _rate_limit_status[idx]['remaining'] > 0:
        # Pick auth with maximum remaining calls
        switch_auth(idx)
    else:
        # Pick auth with minimum reset time
        idx = min(enumerate(_rate_limit_status), key=lambda x: x[1]['reset'])[0]
        sleep_time = _rate_limit_status[idx]['reset'] - int(time.time())
        if sleep_time > 0:
            time.sleep(sleep_time + 5)
        switch_auth(idx)


def remaining_calls(resource, path):
    """ Get the remaining number of calls left for a given API resource """
    assert isinstance(resource, str) and isinstance(path, str)
    result = api.rate_limit_status()['resources'][resource][path]['remaining']
    print(result)
    return result

# Load all the OAuth handlers
auths = []
credentialsFile = open('twitter_credentials.csv', 'r')
credentialsReader = csv.DictReader(credentialsFile)
for cred in credentialsReader:
    auth = tweepy.OAuthHandler(cred['consumer_key'], cred['consumer_secret'])
    auth.set_access_token(cred['access_token'], cred['access_secret'])
    auths.append(auth)
if not auths:
    raise ValueError('No OAuth handlers available.')
print('Imported %s twitter credentials' % len(auths))

# Load the Twitter API
api = tweepy.API(auths[0], retry_count=3, retry_delay=5,
                 retry_errors={401, 404, 500, 503})
api.auths = list(auths)
api.auth_idx = 0
auths = None

################################################
# COLLECT TWEETS
################################################


def list_tweets(cursor, resource, path):
    """ Iterate over the tweets with a cursor """
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            handle_rate_limit(resource, path)


def get_all_tweets_of_user(screen_name):
    """ Get up to all (3240 recent) tweets of given screen name """
    assert isinstance(screen_name, str)

    # Resource from which we want to collect tweets
    resource, path = 'statuses', '/statuses/user_timeline'

    # initialize a list to hold all the tweets
    alltweets = []

    for page in list_tweets(
            tweepy.Cursor(api.user_timeline, screen_name=screen_name, count=200).pages(), resource, path):
        alltweets.extend(page)
        print("...%s tweets downloaded so far" % len(alltweets))

    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.author.id,
                  tweet.author.name,
                  tweet.id_str,
                  tweet.created_at,
                  tweet.text.encode('utf-8'),
                  tweet.retweet_count,
                  tweet.author.followers_count,
                  tweet.author.friends_count,
                  tweet.author.statuses_count] for tweet in alltweets]

    with open('results/%s_tweets.csv' % screen_name, 'w') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["user_id", "screen_name", "tweet_id",
                         "created_at", "text", "retweet_count",
                         "#followers", "#followings", "#statuses"])
        writer.writerows(outtweets)


def get_all_tweets_of_users(list_of_users):
    """ Get the tweets all given users in list """
    assert isinstance(list_of_users, list) and all(isinstance(elem, str) for elem in list_of_users)
    for user in list_of_users:
        get_all_tweets_of_user(user)


def get_tweets_of_users_in_file():
    """
    Loads a list of users from users.p
    pickle file and get their tweets
    """
    # Get list of users
    users = pickle.load(open("users.p", "rb"))

    # Get their tweets
    get_all_tweets_of_users(users)


def set_users(list_of_users):
    """ Add give users to pickle file """
    assert isinstance(list_of_users, list) and all(isinstance(elem, str) for elem in list_of_users)
    users = list_of_users
    pickle.dump(users, open("users.p", "w"))

set_users(["twitter"])
