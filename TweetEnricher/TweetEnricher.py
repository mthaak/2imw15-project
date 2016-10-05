import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
import string,re

class TweetEnricher:
    """
    Class that adds features to tweets collected
    """
    def __init__(self):
        self.negative_opinion_feature = 0
        self.positive_opinion_feature = 0
        self.opinion_feature = 0
        self.vulgar_words_feature = 0
        self.emoticon_feature = 0
        self.enriched_tweets = []
        self.verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.adjective_tags = ['JJ', 'JJR', 'JJS']
        self.interjection_tags = ['UH']
        self.twitter_tags = ['RT', '@', '#']
        self.verbs = []
        self.punctuation_feature = 0
        self.question_feature = 0
        self.recommendation_feature = 0
        self.intergection_feature = 0
        self.adjective_feature = 0
        self.retweet_position_feature = 0
        self.retweet_feature = 0
        self.hashtag_feature = 0
        self.hashtag_position_feature = 0
        self.reply_position_feature = 0
        self.reply_feature = 0
        self.web_abbreviations_feature = 0
        self.twitter_jargons_feature = 0
        self.tokenizer = TweetTokenizer()
        self.stopset = set(stopwords.words('english'))
        self.negative_opinions = opinion_lexicon.negative()
        self.positive_opinions = opinion_lexicon.positive()
        self.vulgar_words = [line.rstrip('\n') for line in open('../Data/Lists/VulgarWordsList')]
        self.twitter_jargons = [line.rstrip('\n') for line in open('../Data/Lists/TwitterSlangsAndAbbreviations')]
        self.web_abbreviations = [line.rstrip('\n') for line in open('../Data/Lists/WebAcronymns')]
        self.emoticons_list = [line.rstrip('\n') for line in open('../Data/Lists/EmojiList')]


    def tokenize(self,tweet):
        """
        Tokens created using nltk tokenizer
        :param tweet:
        :return: tokens from tweet
        """
        return self.tokenizer.tokenize(tweet)

    def removeStopWords(self,tokens):
        """
        Removes stop words from tokens
        :param tokens:
        :return: tokens without stopwords
        """
        tokens_without_stopwords = [w for w in tokens if not w in self.stopset]
        return tokens_without_stopwords

    def hasNegativeOpinions(self,tokens):
        """
        Checks if negative opinions from nltk corpus present
        :param tokens:
        :return: 1 if negative opinions present. 0 otherwise
        """
        for w in tokens:
            if w in self.negative_opinions:
                return 1
        return 0

    def hasPositiveOpinions(self,tokens):
        '''
        Checks if positive opnions present
        :param tokens:
        :return: 1 if positive opinions present. 0 otherwise
        '''
        for w in tokens:
            if w in self.positive_opinions:
                return 1
        return 0

    def hasVulgarWords(self,tokens):
        '''
        Checks if vulgar words from online list present
        :param tokens:
        :return: 1 if present
        '''
        for w in tokens:
            if w in self.vulgar_words:
                return 1
        return 0

    def hasAbbreviations(self,tokens):
        '''
        Checks if abbreviations from online list present
        :param tokens:
        :return: 1 if present
        '''
        for w in tokens:
            if w in self.web_abbreviations:
                return 1
        return 0

    def hasTwitterJargons(self,tokens):
        '''
        Checks if twitter specific abbreviations and jargons from online list present
        :param tokens:
        :return: 1 if present
        '''
        for w in tokens:
            if w in self.twitter_jargons:
                return 1
        return 0

    def hasEmoticons(self,tokens):
        '''
        Checks if abbreviations from online list present
        :param tokens:
        :return: 1 if present
        '''
        for w in tokens:
            if w in self.emoticons_list:
                return 1
        return 0

    def isInterrogative(self,tokens):
        '''
        Checks if ? present
        :param tokens:
        :return: 1 if present
        '''
        for term in tokens:
            if term in string.punctuation:
                if term == "?":
                    return 1
        return 0

    def isExclamatory(self,tokens):
        '''
        Checks if ! present
        :param tokens:
        :return: 1 if present
        '''
        for term in tokens:
            if term in string.punctuation:
                if term == "!":
                    return 1
        return 0

    def hasHash(self,tokens):
        '''
        Checks if # present
        :param tokens:
        :return: 1 if present, 0 otherwise; 1 if present in the beginning, 0 oherwise
        '''
        for index, term in enumerate(tokens):
           if re.match("#.*", term):
                if 2 * index < len(tokens):
                    return 1,1
                else:
                    return 1,0
        return 0,0

    def hasRT(self,tokens):
        '''
        Checks if RT present
        :param tokens:
        :return: 1 if present, 0 otherwise; 1 if present in the beginning, 0 oherwise
        '''
        for index, term in enumerate(tokens):
           if term == "RT":
                if 2 * index < len(tokens):
                    return 1,1
                else:
                    return 1,0
        return 0,0

    def hasATag(self,tokens):
        '''
        Checks if @ present
        :param tokens:
        :return: 1 if present, 0 otherwise; 1 if present in the beginning, 0 oherwise
        '''
        for index, term in enumerate(tokens):
           if re.match("@.*", term):
                if 2 * index < len(tokens):
                    return 1,1
                else:
                    return 1,0
        return 0,0

    def hasALink(self,tokens):
        """
        Checks if the tweet has a url
        :param tokens:
        :return: 1 if url present
        """
        for w in tokens:
            if re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', w):
                return 1
        return 0
