import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
import string
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer

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
        self.brexit_keywords = [line.rstrip('\n') for line in open('../Data/Lists/BrexitKeywords')]
        self.vulgar_words = [line.rstrip('\n') for line in open('../Data/Lists/VulgarWordsList')]
        self.twitter_jargons = [line.rstrip('\n') for line in open('../Data/Lists/TwitterSlangsAndAbbreviations')]
        self.web_abbreviations = [line.rstrip('\n') for line in open('../Data/Lists/WebAcronymns')]
        self.emoticons_list = [line.rstrip('\n') for line in open('../Data/Lists/EmojiList')]
        self.speech_act_verbs = [line.rstrip('\n') for line in open('../Data/Lists/StemmedSpeechActVerbs')]
        self.verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.n_gram_count_matrix = {}
        self.vectorizer = CountVectorizer(ngram_range=(1, 3), tokenizer=self.tokenizer.tokenize,
                                     stop_words=list(self.stopset) + self.web_abbreviations + list(string.punctuation))
        self.vectorizer_unigram = CountVectorizer(ngram_range=(1, 1), tokenizer=self.tokenizer.tokenize,
                                          stop_words=list(self.stopset) + self.web_abbreviations + list(string.punctuation))


    def tweetFeatures(self):
        self.tweet_features = ["VulgarWords", "Emoticons", "Interrogation",
                               "Exclamation", "Abbreviations", "TwitterJargons", "#", "# position", "@", "@ position", "RT", "RT position", "Link"]
        for w in self.speech_act_verbs:
            self.tweet_features.append(w)
        self.tweet_features.append(["NegativeOpnions", "PositiveOpnions"])
        self.basic_features = self.tweet_features
        for w in self.n_gram_count_matrix:
            self.tweet_features.append(w)
        return self.basic_features, self.tweet_features, self.n_gram_count_matrix

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
        count = 0
        negative = 0
        for w in tokens:
            if w in self.negative_opinions:
                count = count + 1
        if count > 0:
            negative = 1
        return count,negative

    def hasPositiveOpinions(self,tokens):
        '''
        Checks if positive opnions present
        :param tokens:
        :return: 1 if positive opinions present. 0 otherwise
        '''
        count = 0
        positive=0
        for w in tokens:
            if w in self.positive_opinions:
                count = count + 1
        if count > 0:
            positive = 1
        return count,positive

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

    def collectNGramFeatures(self,tweet):
        '''
        Collects the n-grams present in tweet that also occur through out the document more than 5 times
        :param tweet:
        :return: a count matrix
        '''

        # initializing matrix to all 0s
        tweet_n_gram_count_dict = {}
        for item in self.n_gram_count_matrix:
            tweet_n_gram_count_dict[item]=0

        self.vectorizer.fit_transform(tweet)
        feature_names = self.vectorizer.get_feature_names()
        for term in nltk.pos_tag(feature_names):
            # Remove proper nouns
            if term[1] == "NNP" or term[1] == "NNPS":
                feature_names.remove(term[0])

        # Remove potential Brexit keywords
        for i in self.brexit_keywords:
            if i in feature_names:
                feature_names.remove(i)

        #If n grams collected present in n grams for whole document too, set 1 for that n gram in matrix row corresponding to tweet
        for n_gram in self.n_gram_count_matrix.keys():
            if n_gram in feature_names:
                tweet_n_gram_count_dict[n_gram] = 1
        return tweet_n_gram_count_dict

    def returnUnigramMatrix(self,document):
        '''
        Reurns unigrams in document- done to fetch Brexit related keywords
        '''
        X = self.vectorizer_unigram.fit_transform(document)
        feature_names = self.vectorizer_unigram.get_feature_names()
        term_freqs = X.sum(axis=0).A1
        unigram_count_matrix = dict(zip(feature_names, term_freqs))

        return unigram_count_matrix

    def createNGramCountMatrix(self,document):
        '''
        Creates a n-gram count matrix of all the tweets. N-grams that have no common nouns and occur more than 5 times
        :param document:
        :return: a count matrix
        '''
        X = self.vectorizer.fit_transform(document)
        feature_names = self.vectorizer.get_feature_names()
        for term in nltk.pos_tag(feature_names):
            # Remove proper nouns
            if term[1] == "NNP" or term[1] == "NNPS":
                feature_names.remove(term[0])

        # Remove potential Brexit keywords
        for i in self.brexit_keywords:
            if i in feature_names:
                feature_names.remove(i)

        term_freqs = X.sum(axis=0).A1
        self.n_gram_count_matrix = dict(zip(feature_names, term_freqs))

        for w in self.n_gram_count_matrix.copy():
          #keep only those n-grams that have a frequency > 5
            if(self.n_gram_count_matrix.get(w) < 5):
                self.n_gram_count_matrix.pop(w)
        return self.tweetFeatures()

    def enrichTweets(self, tweet):
        """
        Features added to a row of data.
        :param row:
        :param tweet:
        :return: Returns a row with features added
        """
        row=[]
        #tokenize tweet
        tokens = self.tokenize(tweet)
        # remove stop words
        tokens = self.removeStopWords(tokens)
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

        basic_row = row

        # add positive and negative opinion feature
        pos_count, pos_presence = self.hasNegativeOpinions(tokens)
        neg_count, neg_presence = self.hasPositiveOpinions(tokens)

        # counts
        row.append(pos_count)
        row.append(neg_count)

        # boolean- positive/negative opinion feature
        basic_row.append(pos_presence)
        basic_row.append(neg_presence)

        #binary features with n grams
        binary_ngrams_row = basic_row

        # n grams from tweet
        n_gram_feature_dict = self.collectNGramFeatures(tweet)
        for w in n_gram_feature_dict:
            row.append(n_gram_feature_dict.get(w))
            binary_ngrams_row.append(n_gram_feature_dict.get(w))

        return row, basic_row, binary_ngrams_row
