import tweepy
import time
import csv
import pickle
import re
import os

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


def load_auth_handlers_from_file(filename):
    """
    Load all the OAuth handlers
    :return: list of OAuth handlers
    """
    auths = []
    credentials_file = open(filename, 'r')
    credentials_reader = csv.DictReader(credentials_file)
    for cred in credentials_reader:
        auth = tweepy.OAuthHandler(cred['consumer_key'], cred['consumer_secret'])
        auth.set_access_token(cred['access_token'], cred['access_secret'])
        auths.append(auth)
    if not auths:
        raise ValueError('No OAuth handlers available.')
    print('Imported %s twitter credentials' % len(auths))
    return auths

# Load the Twitter API
auths = load_auth_handlers_from_file('twitter_credentials.csv')
api = tweepy.API(auths[0], retry_count=3, retry_delay=5,
                 retry_errors={401, 404, 500, 503})
api.auths = list(auths)
api.auth_idx = 0
auths = None

################################################
# COLLECT TWEETS
################################################


def set_users(list_of_users):
    """ Add give users to pickle file """
    assert isinstance(list_of_users, list) and (all(isinstance(elem, str) for elem in list_of_users))
    users = list_of_users
    pickle.dump(users, open("users.p", "wb"))


def get_users():
    """ Get list of users """
    users = pickle.load(open("users.p", "rb"))
    return users


def cursor_iterator(cursor, resource, path):
    """ Iterator for tweepy cursors """
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError as e:
            print(e.reason)
            handle_rate_limit(resource, path)
        except Exception:
            raise StopIteration


def check_keyword(s, key):
    """ Check if keyword exists in string """
    return bool(re.search(key, s, re.IGNORECASE))


def get_all_tweets_of_user(screen_name, keywords=set()):
    """ Get all (max 3240 recent) tweets of given screen name """
    assert isinstance(screen_name, str)
    assert isinstance(keywords, set) and all(isinstance(k, str) for k in keywords)

    # Resource from which we want to collect tweets
    resource, path = 'statuses', '/statuses/user_timeline'

    # initialize a list to hold all the tweets
    alltweets = []

    for page in cursor_iterator(
            tweepy.Cursor(api.user_timeline, screen_name=screen_name, count=200).pages(), resource, path):
        alltweets.extend(page)
        print("...%s tweets downloaded so far" % len(alltweets))

    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.id_str,
                  tweet.text.replace('\n', ' ').replace('\r', ''),
                  tweet.created_at,
                  tweet.retweet_count,
                  tweet.author.id,
                  tweet.author.name,
                  tweet.author.followers_count,
                  tweet.author.friends_count,
                  tweet.author.statuses_count,
                  [k for k in keywords if check_keyword(tweet.text, k)],
                  [hashtag['text'] for hashtag in tweet.entities['hashtags']],
                  [url['expanded_url'] for url in tweet.entities['urls']]] for tweet in alltweets]

    with open(os.path.join('results', '%s_tweets.csv' % screen_name), 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["tweet_id", "text", "created_at", "retweet_count",
                         "user_id", "screen_name", "#followers", "#followings",
                         "#statuses", "keywords", "hashtags", "urls"])
        writer.writerows(outtweets)


def get_all_tweets_of_users(list_of_users, keywords=set()):
    """ Get the tweets all given users in list """
    assert isinstance(list_of_users, list) and all(isinstance(elem, str) for elem in list_of_users)
    for user in list_of_users:
        print('Getting tweets for %s' % user)
        get_all_tweets_of_user(user, keywords)


def get_friends_of_user(screen_name):
    """ Get all friends of the given user
    :param screen_name: Twitter screen name of the given user
    :return: List of all friends of given user
    """
    assert isinstance(screen_name, str)

    # Resource from which we want to collect tweets
    resource, path = 'friends', '/friends/list'

    # initialize a list to hold all the friends screen names
    users = []

    for page in cursor_iterator(
            tweepy.Cursor(api.friends, screen_name=screen_name, count=200).pages(), resource, path):
        users.extend(page)
        print('...%s friends found so far' % len(users))

    # transform the tweepy friends into a 2D array that will populate the csv
    outfriends = [[screen_name,
                   user.id_str,
                   user.screen_name,
                   user.followers_count,
                   user.friends_count,
                   user.listed_count,
                   user.statuses_count] for user in users]

    with open(os.path.join('results', '%s_friends.csv' % screen_name), 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["user_screen_name", "friend_id", "friend_screen_name",
                         "friends_#followers", "friends_#followings", "friends_#listed",
                         "friends_#statuses"])
        writer.writerows(outfriends)

    return [user.screen_name for user in users]


def get_friends_of_users(list_of_users):
    """ Get the tweets all given users in list """
    assert isinstance(list_of_users, list) and all(isinstance(elem, str) for elem in list_of_users)
    for user in list_of_users:
        print('Getting friends of %s' % user)
        get_friends_of_user(user)


def check_query(s):
    """ Checks for common search API query keywords """
    return (check_keyword(s, 'from:')
            or check_keyword(s, 'to:')
            or check_keyword(s, 'list:')
            or check_keyword(s, 'filter:')
            or check_keyword(s, 'url:')
            or check_keyword(s, 'since:')
            or check_keyword(s, 'until:')
            or s == 'OR' or s == '"'
            or s == '#' or s == '?'
            or s == ':)' or s == ':('
            or s[0] == '-' or s[0] == '@' or s[0] == '#')


def search_tweets(query, output_filename):
    assert isinstance(query, str) and isinstance(output_filename, str)

    # Get all the relevant keywords from the query
    import shlex
    keywords = set(s.replace('(', '').replace(')', '') for s in shlex.split(query) if not check_query(s))
    print('keywords: ', keywords)

    from datetime import datetime
    time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Resource from which we want to collect tweets
    resource, path = 'search', '/search/tweets'

    # initialize a list to hold all the tweets
    alltweets = []

    for page in cursor_iterator(
            tweepy.Cursor(api.search, q=query, count=200).pages(), resource, path):
        alltweets.extend(page)
        print("...%s tweets downloaded so far" % len(alltweets))
        if len(alltweets) > 3000:
            break

    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.id_str,
                  tweet.text.replace('\n', ' ').replace('\r', ''),
                  tweet.created_at,
                  tweet.retweet_count,
                  tweet.author.id,
                  tweet.author.name,
                  tweet.author.followers_count,
                  tweet.author.friends_count,
                  tweet.author.statuses_count,
                  [k for k in keywords if check_keyword(tweet.text, k)],
                  [hashtag['text'] for hashtag in tweet.entities['hashtags']],
                  [url['expanded_url'] for url in tweet.entities['urls']]] for tweet in alltweets]

    with open(os.path.join('results', '%s_%s_tweets.csv' % (output_filename, time)),
              mode='w', newline='', encoding='utf8') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["tweet_id", "text", "created_at", "retweet_count",
                         "user_id", "screen_name", "#followers", "#followings",
                         "#statuses", "keywords", "hashtags", "urls"])
        writer.writerows(outtweets)

if __name__ == "__main__":
    # Set users from whom to get tweets
    # set_users(list_of_users=['vote_leave', 'BorisJohnson', 'David_Cameron',
    #                          'Nigel_Farage', 'michaelgove', 'George_Osborne'])

    # Load users
    # users = get_users()

    # Get tweets
    # get_all_tweets_of_users(users, keywords=["people", "twitter"])

    # Get friends
    # get_friends_of_users(users)

    # Search tweets on keywords
    query = '(britain eu) OR referendum OR brexit OR #voteleave OR #votestay'
    search_tweets(query, 'search')
