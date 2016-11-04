import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
import string
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
import math
import csv

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
        self.pos_emoticons_list = [line.rstrip('\n') for line in open('../Data/Lists/PositiveEmojiList')]
        self.neg_emoticons_list = [line.rstrip('\n') for line in open('../Data/Lists/NegativeEmojiList')]
        self.first_person_pronouns = [line.rstrip('\n') for line in open('../Data/Lists/FirstPersonPronouns')]
        self.speech_act_verbs = [line.rstrip('\n') for line in open('../Data/Lists/StemmedSpeechActVerbs')]
        self.trusted_domains = [line.rstrip('\n') for line in open('../Data/Lists/TrustedDomains')]
        self.verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.n_gram_count_matrix = {}
        self.vectorizer = CountVectorizer(ngram_range=(1, 3), tokenizer=self.tokenizer.tokenize,
                                          stop_words=list(self.stopset)
                                                     + self.web_abbreviations
                                                     + list(string.punctuation)
                                                     + ["…", "...", "..", ")", "(", "-->", "->", ">>", "#", "RT", "@"])
        self.vectorizer_unigram = CountVectorizer(ngram_range=(1, 1), tokenizer=self.tokenizer.tokenize,
                                                  stop_words=list(self.stopset)
                                                             + self.web_abbreviations
                                                             + list(string.punctuation))


    def tweetFeatures(self):
        self.tweet_features = ["VulgarWords", "Emoticons", "Interrogation", "Exclamation", "Abbreviations",
                               "TwitterJargons", "#", "# position", "@", "@ position", "RT", "RT position", "Link"]
        for w in self.speech_act_verbs:
            self.tweet_features.append(w)
        self.tweet_features.append(["NegativeOpnions", "PositiveOpnions","NumbersPresent","Non-Ascii Characters","GoodLink"])
        self.basic_features = self.tweet_features
        for w in self.n_gram_count_matrix:
            self.tweet_features.append(w)
        return self.basic_features, self.tweet_features, self.n_gram_count_matrix

    def tokenize(self,tweet):
        """
        Tokens created using nltk tokenizer
        :return: tokens from tweet
        """
        return self.tokenizer.tokenize(tweet)

    def removeStopWords(self,tokens):
        """
        Removes stop words from tokens
        :return: tokens without stopwords
        """
        tokens_without_stopwords = [w for w in tokens if w not in self.stopset]
        return tokens_without_stopwords

    def hasNegativeOpinions(self,tokens):
        """
        Checks if negative opinions from nltk corpus present
        :return: 1 if negative opinions present. 0 otherwise
        """
        count = 0
        negative = 0
        negative_ops = [x.lower() for x in self.negative_opinions]
        for w in tokens:
            if w.lower() in negative_ops:
                count += 1
        if count > 0:
            negative = 1
        return count, negative

    def hasPositiveOpinions(self,tokens):
        '''
        Checks if positive opnions present
        :return: 1 if positive opinions present. 0 otherwise
        '''
        count = 0
        positive = 0
        positive_ops = [x.lower() for x in self.positive_opinions]
        for w in tokens:
            if w.lower() in positive_ops:
                count += 1
        if count > 0:
            positive = 1
        return count, positive

    def hasVulgarWords(self,tokens):
        '''
        Checks if vulgar words from online list present
        :return: 1 if present
        '''
        for w in tokens:
            if w.lower() in [x.lower() for x in self.vulgar_words]:
                return 1
        return 0

    def hasAbbreviations(self,tokens):
        '''
        Checks if abbreviations from online list present
        :return: 1 if present
        '''
        for w in tokens:
            if w.lower() in [x.lower() for x in self.web_abbreviations]:
                return 1
        return 0

    def hasTwitterJargons(self,tokens):
        '''
        Checks if twitter specific abbreviations and jargons from online list present
        :return: 1 if present
        '''
        for w in tokens:
            if w in self.twitter_jargons:
                return 1
        return 0

    def hasEmoticons(self,tokens):
        '''
        Checks if abbreviations from online list present
        :return: 1 if present
        '''
        for w in tokens:
            if w in self.emoticons_list:
                return 1
        return 0

    def isInterrogative(self,tokens):
        '''
        Checks if ? present
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
        :return: 1 if present, 0 otherwise; 1 if present in the beginning, 0 oherwise
        '''
        for index, term in enumerate(tokens):
            if re.match("#\w*", term):
                if 2 * index < len(tokens):
                    return 1,1
                else:
                    return 1,0
        return 0,0

    def hasRT(self,tokens):
        '''
        Checks if RT present
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
        :return: 1 if present, 0 otherwise; 1 if present in the beginning, 0 oherwise
        '''
        for index, term in enumerate(tokens):
            if re.match("@\w*", term):
                if 2 * index < len(tokens):
                    return 1,1
                else:
                    return 1,0
        return 0,0

    def hasALink(self,tokens):
        """
        Checks if the tweet has a url
        :return: 1 if url present
        """
        for w in tokens:
            if re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', w):
                return 1
        return 0

    def hasManyNumbers(self,tokens):
        """
        Checks if the tweet many numbers
        :return: 1 if more than 5 numbers present
        """
        count = 0
        for w in tokens:
            if re.match('[\d]+',w):
               count = count + 1
        if count > 2:
            return 1
        else:
            return 0

    def hasManyNonAscii(self, tokens):
        """
        Checks if the tweet has non-ascii characters
        :return: 1 if more than 5 such present
        """
        count = 0
        for token in tokens:
            if re.match('[^\x00-\x7F]',token):
                count = count + 1
        if count > 5:
            return 1
        else:
            return 0

    def hasLinksToReputedDomains(self, url_string):
        """
        Checks if the tweet links to some trusted pages
        :return: 1 if more any such link present
        """
        for i in self.trusted_domains:
            pattern = re.compile(i+'.*')
            if re.findall(pattern,url_string):
                return 1
        return 0

    def hasSpeechActVerbs(self,tokens):
        '''
        Returns 265 binary features for the speech act verbs
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
        :return: a count matrix
        '''

        # initializing matrix to all 0s
        tweet_n_gram_count_dict = {}

        for gram in self.n_gram_count_matrix:
            tweet_n_gram_count_dict[gram] = 0
            p = re.compile(re.escape(gram))
            if re.findall(p, str(tweet)):
                tweet_n_gram_count_dict[gram] = 1

        return tweet_n_gram_count_dict

    def returnUnigramMatrix(self,collection):
        '''
        Reurns unigrams in document- done to fetch Brexit related keywords
        '''
        X = self.vectorizer_unigram.fit_transform(collection)
        feature_names = self.vectorizer_unigram.get_feature_names()
        term_freqs = X.sum(axis=0).A1
        unigram_count_matrix = dict(zip(feature_names, term_freqs))

        return unigram_count_matrix

    def createNGramCountMatrix(self,collection,SA_tagged_collection):
        '''
        Creates a n-gram count matrix of all the tweets. N-grams that have no common nouns and occur more than 5 times
        :return: a count matrix
        '''
        X = self.vectorizer.fit_transform(collection)
        feature_names = self.vectorizer.get_feature_names()
        print("Size of n grams =  %d " % len(feature_names))

        #Remove n-grams with #,@,RT, only numbers
        for i in feature_names:
            if re.findall('#.*', i):
                feature_names.remove(i)
            elif re.findall('[\d]+', i):
                feature_names.remove(i)
            elif re.findall('@.*', i):
                feature_names.remove(i)
            elif re.findall('RT.*', i):
                feature_names.remove(i)
            elif re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',i):
                feature_names.remove(i)

        # Remove potential Brexit keywords
        for i in feature_names:
            for j in i.split():
                if j in self.brexit_keywords:
                    feature_names.remove(i)
                    break


        for term in nltk.pos_tag(feature_names):
            # Remove proper nouns
            if term[1] == "NNP" or term[1] == "NNPS" or term[1] == "NN":
                feature_names.remove(term[0])


        #Find entropy of each n-gram
        entropy = {}
        for item in feature_names:
            entropy[item] = 0
        total = len(SA_tagged_collection)
        occurrences = {}
        tagged_tweets= {}
        for gram in SA_tagged_collection:
            if SA_tagged_collection.get(gram)[1] != '':
                tagged_tweets[gram] = SA_tagged_collection.get(gram)

        for item in feature_names:
            p = re.compile(re.escape(item))
            for gram in tagged_tweets:
                occurrences[tagged_tweets.get(gram)[1]] = 0
            for gram in tagged_tweets:
                times_in_tweet = len(re.findall(p, str(tagged_tweets.get(gram)[0])))
                occurrences[tagged_tweets.get(gram)[1]] += times_in_tweet
            for gram in tagged_tweets:
                entropy[item] += -(occurrences[tagged_tweets.get(gram)[1]]/total)*math.log((occurrences[tagged_tweets.get(gram)[1]]+1)/total)

        # n-gram entropies
        with open('../Data/test/entropy_raw.csv', 'w+', newline='', encoding='utf8') as out_file:
            writer = csv.writer(out_file)
            for w in sorted(entropy, key=entropy.get):
                writer.writerow([w, entropy.get(w)])

        # # USING good n-grams generated before by current commented sections entropy threshold 0.2
        # feature_names = [line.rstrip('\n') for line in open('../Data/test/entropy_lt_0.5_counts', encoding='utf8')]

        term_freqs = X.sum(axis=0).A1
        self.n_gram_count_matrix = dict(zip(feature_names, term_freqs))

        for w in self.n_gram_count_matrix.copy():
          #keep only those n-grams that have a frequency > 5
            if(self.n_gram_count_matrix.get(w) < 5):
                self.n_gram_count_matrix.pop(w)

        print("Reduced Size of n grams = %d " % len(entropy))

        # Normalize by dividing by log of number of occurences on n-gram in collection
        for gram in entropy.copy():
            if gram in self.n_gram_count_matrix.keys():
                entropy[gram] = entropy[gram]/(math.log(self.n_gram_count_matrix.get(gram)))
            else:
                entropy.pop(gram)

        print("Reduced Size of n grams = %d " % len(entropy))

        # n-gram entropies
        with open('../Data/test/entropy.csv', 'w+', newline='', encoding='utf8') as out_file:
            writer = csv.writer(out_file)
            for w in sorted(entropy, key=entropy.get):
                writer.writerow([w, entropy.get(w)])

        return self.tweetFeatures()

    def speechActTagTweet(self,tweet):
        tokens = self.tokenize(tweet)
        pos_list = nltk.pos_tag(tokens)
        tag = ''
        for entry in pos_list:
            if entry[1] in self.verb_tags:  # check if the tokens that are verbs are speech act verbs
                verb = self.stemmer.stem(entry[0].lower())
                if verb in self.speech_act_verbs:
                    tag = verb
                    break
        return tag

    def speechActTagCollection(self, collection):
        self.speech_act_tags = {}
        for tweet in collection:
            self.speech_act_tags[tweet] = (collection.get(tweet), self.speechActTagTweet(collection.get(tweet)))
        return self.speech_act_tags

    def enrichTweets(self, tweet, urls):
        """
        Features added to a row of data.
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

        #more than 5 numbers in tweet
        row.append(self.hasManyNumbers(tokens))

        #presence of non-ascii characters
        row.append(self.hasManyNonAscii(tokens))

        #presence of links to trusted domains
        row.append(self.hasLinksToReputedDomains(urls))

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

    def enrichTweetsWithNgrams(self, tweet):
        '''
        Returns only n gram features for tweet
        '''
        row = []
        n_gram_feature_dict = self.collectNGramFeatures(tweet)
        for w in n_gram_feature_dict:
            row.append(n_gram_feature_dict.get(w))
        return row

    def hasPositiveEmoticons(self, tokens):
        '''
        Checks if abbreviations from online list present
        :return: 1 if present
        '''
        for w in tokens:
            if w in self.pos_emoticons_list:
                return 1
        return 0

    def hasNegativeEmoticons(self, tokens):
        '''
        Checks if abbreviations from online list present
        :return: 1 if present
        '''
        for w in tokens:
            if w in self.neg_emoticons_list:
                return 1
        return 0

    def sentiment(self, text):
        tokens = self.tokenize(text)
        tokens = [w for w in tokens if not w in set(stopwords.words('english') + self.web_abbreviations + list(string.punctuation))]

        positive_count, is_positive = self.hasPositiveOpinions(tokens)
        negative_count, is_negative = self.hasNegativeOpinions(tokens)

        positive_count += self.hasPositiveEmoticons(tokens)
        negative_count += self.hasNegativeEmoticons(tokens)

        positive_percentage = positive_count / len(tokens)
        negative_percentage = negative_count / len(tokens)

        return positive_percentage, negative_percentage

    def hasFirstPersonPronouns(self, tokens):
        for w in tokens:
            if w in self.first_person_pronouns:
                return 1
        return 0
