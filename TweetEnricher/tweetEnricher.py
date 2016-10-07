import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
import string
import re
from nltk.stem.porter import *

class TweetEnricher:
    """
    Class that adds features to tweets collected
    """
    def __init__(self):
        self.tokenizer = TweetTokenizer()
        self.stemmer = PorterStemmer()
        self.stopset = set(stopwords.words('english'))
        self.negative_opinions = opinion_lexicon.negative()
        self.positive_opinions = opinion_lexicon.positive()
        self.vulgar_words = [line.rstrip('\n') for line in open('../Data/Lists/VulgarWordsList')]
        self.twitter_jargons = [line.rstrip('\n') for line in open('../Data/Lists/TwitterSlangsAndAbbreviations')]
        self.web_abbreviations = [line.rstrip('\n') for line in open('../Data/Lists/WebAcronymns')]
        self.emoticons_list = [line.rstrip('\n') for line in open('../Data/Lists/EmojiList')]
        self.speech_act_verbs = [line.rstrip('\n') for line in open('../Data/Lists/StemmedSpeechActVerbs')]
        self.verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

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

    def hasSpeechActVerbs(self,tokens):
        '''
        Returns 265 binary features for the speech act verbs
        :param tokens:
        :return: a dictionary
        '''
        speech_act_feature_dict = {}        # dict stores the 265 features
        for w in self.speech_act_verbs:     # initializing
            speech_act_feature_dict[w] = 0
        pos_list = nltk.pos_tag(tokens)
        for entry in pos_list:
            if entry[1] in self.verb_tags:   # check if the tokens that are verbs are speech act verbs
                verb = self.stemmer.stem(entry[0].lower())
                if verb in self.speech_act_verbs:
                    speech_act_feature_dict[verb] = 1
        return speech_act_feature_dict


    def enrichTweets(self, row, tokens):
        """
        Features added to a row of data.
        :param row:
        :param tokens:
        :return: Returns a row with features added
        """
        # remove stop words
        tokens = self.removeStopWords(tokens)
        # add negative opinion feature
        row.append(self.hasNegativeOpinions(tokens))
        # add positive opinions feature
        row.append(self.hasPositiveOpinions(tokens))
        # add  Vulgar words feature
        row.append(self.hasVulgarWords(tokens))
        # add Emoticons feature
        row.append(self.hasEmoticons(tokens))
        # add Interrogation feature
        row.append(self.isInterrogative(tokens))
        # add Exclamation feature
        row.append((self.isExclamatory(tokens)))
        # add Abbreviations feature
        row.append(self.hasAbbreviations(tokens))
        # add twitter jargons feature
        row.append(self.hasTwitterJargons(tokens))
        # add Twiiter specific characters' features- presence and position
        presence, position = self.hasHash(tokens)
        row.append(presence)
        row.append(position)

        presence, position = self.hasATag(tokens)
        row.append(presence)
        row.append(position)

        presence, position = self.hasRT(tokens)
        row.append(presence)
        row.append(position)

        # add URL presence feature
        row.append(self.hasALink(tokens))
        #add speech act verbs features
        sac_feature_dict = self.hasSpeechActVerbs(tokens)
        for verb in self.speech_act_verbs:
              row.append(sac_feature_dict.get(verb))

        return row
