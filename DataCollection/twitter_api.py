import tweepy
import time
import csv
import pickle
import re
import os
from DataCollection.utils import chunks, sleep_with_countdown
from datetime import datetime
from progressbar import ProgressBar
from sys import stderr


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
    print('\t> Handling rate limit', file=stderr)

    # Get rate limit status of all OAuth credentials
    _rate_limit_status = []
    for auth in api.auths:
        api.auth = auth
        result = api.rate_limit_status()['resources'][resource][path]
        _rate_limit_status.append(result)

    # IF maximum remaining calls in all auths is 0
    # THEN sleep till reset time.
    idx = max(enumerate(_rate_limit_status), key=lambda x: x[1]['remaining'])[0]
    if _rate_limit_status[idx]['remaining'] == 0:
        # Pick auth with minimum reset time
        idx = min(enumerate(_rate_limit_status), key=lambda x: x[1]['reset'])[0]
        sleep_time = _rate_limit_status[idx]['reset'] - int(time.time())
        if sleep_time > 0:
            sleep_with_countdown(sleep_time + 5)

    # Pick auth with maximum remaining calls
    switch_auth(idx)


def remaining_calls(resource, path):
    """ Get the remaining number of calls left for a given API resource """
    assert isinstance(resource, str) and isinstance(path, str)
    result = api.rate_limit_status()['resources'][resource][path]['remaining']
    print('> Remaining calls for', path, ':', result, file=stderr)
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
    # First check to make sure enough calls are available
    if remaining_calls(resource, path) == 0:
        handle_rate_limit(resource, path)
    err_count = 0

    while True:
        try:
            yield cursor.next()
            remaining = int(api.last_response.headers['x-rate-limit-remaining'])
            if remaining == 0:
                handle_rate_limit(resource, path)
        except tweepy.RateLimitError as e:
            print(e.reason)
            err_count += 1
            if err_count > 1:
                break
            else:
                handle_rate_limit(resource, path)
        except tweepy.error.TweepError as e:
            print(e.response)
            print(e.api_code)
            err_count += 1
            if err_count > 1:
                break
            elif e.api_code == 429:
                handle_rate_limit(resource, path)
        except Exception as e:
            print(e)
            break
        else:
            err_count = 0


def check_keyword(s, key):
    """ Check if keyword exists in string """
    return bool(re.search(key, s, re.IGNORECASE))


def get_tweets_of_user(screen_name, count=-1, keywords=set(), save_to_csv=True):
    """ Get all (max 3240 recent) tweets of given screen name """
    assert isinstance(screen_name, str)
    assert isinstance(keywords, set) and all(isinstance(k, str) for k in keywords)
    assert isinstance(count, int) and count >= -1
    assert isinstance(save_to_csv, bool)

    # Resource from which we want to collect tweets
    resource, path = 'statuses', '/statuses/user_timeline'

    # initialize a list to hold all the tweets
    results = []

    try:
        for page in cursor_iterator(
                tweepy.Cursor(api.user_timeline, screen_name=screen_name,
                              count=200, include_rts=True).pages(), resource, path):
            results.extend(page)
            print("...%s results downloaded so far" % len(results))
            if 0 < count <= len(results):
                break
    except KeyboardInterrupt:
        pass

    # transform the tweepy tweets into a 2D array that will populate the csv
    filtered_results = [[tweet.id_str,
                         tweet.text.replace('\n', ' ').replace('\r', ''),
                         tweet.created_at,
                         tweet.retweet_count,
                         tweet.favorite_count,
                         1 if tweet.in_reply_to_user_id is not None else 0,
                         tweet.in_reply_to_user_id if tweet.in_reply_to_user_id is not None else 0,
                         tweet.in_reply_to_status_id_str if tweet.in_reply_to_status_id_str is not None else 0,
                         tweet.author.id,
                         tweet.author.screen_name,
                         tweet.author.created_at,
                         tweet.author.followers_count,
                         tweet.author.friends_count,
                         tweet.author.statuses_count,
                         tweet.author.listed_count,
                         tweet.author.favourites_count,
                         1 if tweet.author.verified else 0,
                         [k for k in keywords if check_keyword(tweet.text, k)],
                         [hashtag['text'] for hashtag in tweet.entities['hashtags']],
                         [url['expanded_url'] for url in tweet.entities['urls']]] for tweet in results]

    features = ["tweet_id", "text", "created_at", "#retweets", "#favorites", "is_reply", "reply_to_user_id",
                "reply_to_tweet_id", "user_id", "screen_name", "user_created_at", "#followers",
                "#followings", "#statuses", '#listed', "#favourites", "verified", "keywords",
                "hashtags", "urls"]
    if len(filtered_results) > 0 and len(features) != len(filtered_results[0]):
        print('Features are not aligned to the result!')

    if save_to_csv and len(filtered_results) > 0:
        with open(os.path.join('results', '%s_tweets.csv' % screen_name), 'w', newline='', encoding='utf8') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(features)
            writer.writerows(filtered_results)

    return features, filtered_results


def get_all_tweets_of_users(list_of_users, nr_of_tweets=-1, keywords=set()):
    """ Get the tweets all given users in list """
    assert isinstance(list_of_users, list) and all(isinstance(elem, str) for elem in list_of_users)
    for user in list_of_users:
        print('Getting tweets for %s' % user)
        get_tweets_of_user(user, count=nr_of_tweets, keywords=keywords)


def get_friends_of_user(screen_name, save_to_csv=True):
    """ Get all friends of the given user
    :param screen_name: Twitter screen name of the given user
    :param save_to_csv:
    :return: List of all friends of given user
    """
    assert isinstance(screen_name, str)
    assert isinstance(save_to_csv, bool)

    # Resource from which we want to collect tweets
    resource, path = 'friends', '/friends/list'

    # initialize a list to hold all the friends screen names
    results = []

    for page in cursor_iterator(
            tweepy.Cursor(api.friends, screen_name=screen_name, count=200).pages(), resource, path):
        results.extend(page)
        print('...%s results found so far' % len(results))

    # transform the tweepy friends into a 2D array that will populate the csv
    filtered_results = [[screen_name,
                         user.id_str,
                         user.screen_name,
                         user.followers_count,
                         user.friends_count,
                         user.listed_count,
                         user.statuses_count] for user in results]

    features = ["user_screen_name", "friend_id", "friend_screen_name", "friends_#followers",
                "friends_#followings", "friends_#listed", "friends_#statuses"]
    if len(filtered_results) > 0 and len(features) != len(filtered_results[0]):
        print('Features are not aligned to the result!')

    if save_to_csv and len(filtered_results) > 0:
        with open(os.path.join('results', '%s_friends.csv' % screen_name), 'w', newline='', encoding='utf8') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(features)
            writer.writerows(filtered_results)

    return features, filtered_results


def get_friends_ids_of_user(screen_name, count=5000, save_to_csv=True):
    """ Get all friends of the given user
    :param screen_name: Twitter screen name of the given user
    :param count:
    :param save_to_csv:
    :return: List of all friends of given user
    """
    assert isinstance(screen_name, str)
    assert isinstance(count, int)
    assert isinstance(save_to_csv, bool)

    # Resource from which we want to collect tweets
    resource, path = 'friends', '/friends/ids'

    # initialize a list to hold all the friends screen names
    results = []

    for page in cursor_iterator(
            tweepy.Cursor(api.friends_ids, screen_name=screen_name, count=count).pages(), resource, path):
        results.extend(page)
        print('...%s results found so far' % len(results))
        if len(results) >= count:
            break

    # transform the tweepy friends into a 2D array that will populate the csv
    filtered_results = [[screen_name, results]]

    features = ["user_screen_name", "friend_ids"]
    if len(filtered_results) > 0 and len(features) != len(filtered_results[0]):
        print('Features are not aligned to the result!')

    if save_to_csv and len(filtered_results) > 0:
        with open(os.path.join('results', '%s_friends.csv' % screen_name), 'w', newline='', encoding='utf8') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(features)
            writer.writerows(filtered_results)

    return features, filtered_results


def get_friends_of_users(list_of_users):
    """ Get the tweets all given users in list """
    assert isinstance(list_of_users, list) and all(isinstance(elem, str) for elem in list_of_users)
    for user in list_of_users:
        print('Getting friends of %s' % user)
        get_friends_of_user(user)


def get_user_info(screen_name):
    """
    Get user object for given screen_name
    :param screen_name: the user
    :return: User object
    """
    assert isinstance(screen_name, str)
    return api.get_user(screen_name=screen_name)


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


def search_tweets(qry, count=-1, since_id=None, max_id=None, save_to_csv=True):
    assert isinstance(qry, str)
    assert isinstance(max_id, int) or max_id is None
    assert isinstance(since_id, int) or since_id is None
    assert isinstance(count, int) and count >= -1
    assert isinstance(save_to_csv, bool)

    # Get all the relevant keywords from the query
    import shlex
    keywords = set(s.replace('(', '').replace(')', '') for s in shlex.split(qry) if not check_query(s))
    print('keywords: ', keywords)

    # Resource from which we want to collect tweets
    resource, path = 'search', '/search/tweets'

    # initialize a list to hold all the tweets
    results = []

    try:
        for page in cursor_iterator(
                tweepy.Cursor(api.search, q=qry, count=200, lang='en', since_id=since_id,
                              max_id=max_id).pages(), resource, path):
            results.extend(page)
            print("...%s results downloaded so far" % len(results))
            if 0 < count <= len(results):
                break
    except KeyboardInterrupt:
        pass

    # transform the tweepy tweets into a 2D array that will populate the csv
    filtered_results = [[tweet.id_str,
                         tweet.text.replace('\n', ' ').replace('\r', ''),
                         tweet.created_at,
                         tweet.retweet_count,
                         1 if tweet.in_reply_to_user_id is not None else 0,
                         tweet.in_reply_to_user_id if tweet.in_reply_to_user_id is not None else 0,
                         tweet.in_reply_to_status_id if tweet.in_reply_to_status_id is not None else 0,
                         tweet.author.id,
                         tweet.author.screen_name,
                         tweet.author.created_at,
                         tweet.author.followers_count,
                         tweet.author.friends_count,
                         tweet.author.statuses_count,
                         tweet.author.listed_count,
                         tweet.author.favourites_count,
                         1 if tweet.author.verified else 0,
                         [k for k in keywords if check_keyword(tweet.text, k)],
                         [hashtag['text'] for hashtag in tweet.entities['hashtags']],
                         [url['expanded_url'] for url in tweet.entities['urls']]] for tweet in results]

    features = ["tweet_id", "text", "created_at", "retweet_count", "is_reply", "reply_to_user_id",
                "reply_to_tweet_id", "user_id", "screen_name", "user_created_at", "#followers",
                "#followings", "#statuses", '#listed', "#favourites", "verified", "keywords",
                "hashtags", "urls"]
    if len(filtered_results) > 0 and len(features) != len(filtered_results[0]):
        print('Features are not aligned to the result!')

    if save_to_csv and len(filtered_results) > 0:
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join('results', 'search_%s_tweets.csv' % time),
                  mode='w', newline='', encoding='utf8') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(['#query', qry])
            writer.writerow(features)
            writer.writerows(filtered_results)

    return features, filtered_results


def lookup_users(user_ids=None, save_to_csv=True):
    assert user_ids is None or (isinstance(user_ids, list) and all(isinstance(k, int) for k in user_ids))
    # assert screen_names is None or (isinstance(screen_names, list) and all(isinstance(k, str) for k in screen_names))
    # assert (user_ids is None) != (screen_names is None)
    assert isinstance(save_to_csv, bool)

    # Resource from which we want to collect tweets
    resource, path = 'users', '/users/lookup'

    # initialize a list to hold all the tweets
    results = []

    # initialize a generator class for chunking user_ids
    class Obj(object):
        _chunks = chunks(user_ids, 100)

        def next(self):
            data = next(self._chunks)
            if len(data) == 0:
                raise StopIteration
            return api.lookup_users(user_ids=data)

    cursor = Obj()

    try:
        for items in cursor_iterator(cursor, resource, path):
            results.extend(items)
            print("...%s results downloaded so far" % len(results))
    except KeyboardInterrupt:
        pass

    # transform the tweepy friends into a 2D array that will populate the csv
    filtered_results = [[user.id_str,
                         user.screen_name] for user in results]

    features = ['user_id', "screen_name"]
    if len(filtered_results) > 0 and len(features) != len(filtered_results[0]):
        print('Features are not aligned to the result!')

    if save_to_csv and len(filtered_results) > 0:
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join('results', 'users_lookup_%s.csv' % time),
                  mode='w', newline='', encoding='utf8') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(features)
            writer.writerows(filtered_results)

    return features, filtered_results


def get_friends_map_from_tweets(file_name, count):
    """
    Generate a friends map from given tweets dataset. Tweets must contain
    screen name of user author.
    :param file_name:
    :return:
    """
    assert isinstance(file_name, str)
    assert isinstance(count, int)

    from DataCollection.utils import read_csv_ignore_comments as read_csv
    df = read_csv(os.path.join('results', file_name))
    if 'screen_name' in df:
        with open(os.path.join('results', os.path.splitext(file_name)[0] + '_user_friends_map.csv'),
                  'w', newline='', encoding='utf8') as f:
            writer = csv.writer(f, delimiter="\t")
            bar = ProgressBar()
            users = set()
            for i, user in bar(df['screen_name'].iteritems()):
                if user in users:
                    continue
                features, results = get_friends_ids_of_user(user, count=count, save_to_csv=False)
                if i == 0:
                    writer.writerow(features)
                if len(results) > 0:
                    writer.writerows(results)
                users.add(user)
    else:
        raise TypeError('Given dataframe must be a tweets dataset with screen_name per tweet.')


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

    # Remaining calls
    # resource, path = 'statuses', '/statuses/user_timeline'
    # remaining_calls(resource, path)

    # Search tweets on keywords
    # since_id = most recent tweet id
    # max_id = oldest retrieved tweet id - 1
    query = '(britain eu) ' \
            'OR ((uk OR britain OR ukip) referendum) ' \
            'OR brexit ' \
            'OR #voteleave ' \
            'OR #votestay ' \
            'OR #EUreferendum ' \
            'OR #StrongerIn ' \
            'OR #Euref ' \
            'OR #Remain ' \
            'OR #voteremain'
    # search_tweets(query, nr_of_tweets=20000, since_id=790324301446676480)#, max_id=790314732670640127)
    # remaining_calls('search', '/search/tweets')

    # Get user friends map
    get_friends_map_from_tweets(file_name='search_20161102_211623_tweets.csv', count=300)

    # user_map = {
    #     "111", {1112, 1113},
    #     "1112", {1114, 1113},
    #     "1113", {111}
    # }
    # users = {}
    # users.retweets
    # users.likes
    # "111" in users
