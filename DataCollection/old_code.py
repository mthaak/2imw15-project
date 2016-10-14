def get_all_tweets_of_user(screen_name):
    """ Get up to all (3240 recent) tweets of given screen name """
    # Resource from which we want to collect tweets
    resource, path = 'statuses', '/statuses/user_timeline'

    # initialize a list to hold all the tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    try:
        new_tweets = api.user_timeline(screen_name=screen_name, count=200)
    except tweepy.RateLimitError:
        handle_rate_limit(resource, path)
        new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    while len(new_tweets) > 0:
        print("getting tweets before %s" % oldest)

        # all subsequent requests use the max_id param to prevent duplicates
        try:
            new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)
        except tweepy.RateLimitError:
            handle_rate_limit(resource, path)
            new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

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