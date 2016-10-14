File tweets.csv
This file uses "tab" as columns separator.
This file is organized as follows (refer to the columns of the dataset as A, B, C, etc...):
A	|B	 |C	|D	  |E	  |F		    |G		 |H		|I	    |J	 |K
keyword	|UserName|UserID|TweetText|TweetID|DateOfPublication|#ofFollowers|#ofFollowings	|#ofStatuses|URLs|Label

Columns meanings:
A: keyword used for the original query to build the dataset
B: The user name of the author of the tweet, it's unique for every user (possible to query user information by the name)
C: User ID of the author of the tweet, it's possible to use instead of the UserName (type long)
D: Message sent by the author, sometimes you can see symbols instead of some special characters (accents, emoticons...), you should
 manage this implementing an easy method to avoid errors while reading the text
E: TweetID (type long), it's not possible to use this data to retrieve new information with the twitter API because you're not allowed to trace tweets older than 1 week. By the way it's a useful information to discriminate tweets during data analysis.
F: Date and time of the post
G, H, I : some information related to the author
J: if URLs are attached to the post, you can see it in the last column; all URLs terminate with a _ (underscore) and if there is more than 1 URL, they are separated through this symbol. (to discriminate different URLs you can try to find "_http" to identify the start of the second URL and so on). Some of the URLs are shortened by twitter, there are a lot of tools to expand URLs if needed.
K: Tweet label-->R = rumour, NR = non rumour, U = unknown

File retweets.csv
This file uses "tab" as columns separator.
This file has been built from the tweets.csv file, saving in retweets.csv all the microblogs that were retweets. 
Columns meanings:
Columns A to F are the same as the tweets.csv file;
Columns G to J represent the original post information:
G: UserName of the original author (the one that posted the retweeted microblog for the first time)
H: User ID of the original author
I: Original tweet text (must be the same as the one in column D, but without the "RT @..." incipit
J: Tweet ID of the original microblog

NB: this file doesn't present the column for the labels, because the RTs do actually belong to the tweets.csv file; by the way it's easy to remap the labels by using the tweetID.

File replies.csv
This file uses "tab" as columns separator.
This file is built starting from the tweets.csv file: all the "replies" were identified and from that replies all the tweets member of the conversation were retrieved. Then the tweets members of the reconstructed conversations were saved in this file organized as follows: 
A: keyword
B: Reply UserName
C: Reply UserID
D: Reply tweetID
E: ReplyText
F: Reply Date
G: Replied UserName
H: Replied UserID
I: Replied tweetID
J: Replied text

The replies labels are saved in a different file named 
repliesLabel.csv
This file uses "tab" as columns separator.
The tweets were divided in conversations so that in this file you can find:
ConversationID:
		Replied tweetID | Reply tweetID | Replied tweet Label| Reply tweet Label

Hence it will be necessary to remap the labels with the corresponding tweet belonging to the replies.csv file by using the tweetID. 
