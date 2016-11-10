from nltk.corpus import opinion_lexicon
from nltk.tokenize import treebank
from nltk.corpus import wordnet
from TweetEnricher import tweetEnricher as TW
import csv
class OpinionClassifier:

    def __init__(self):
        self.RATIO = 1.2
        self.pos_lexicon = opinion_lexicon.positive();
        self.neg_lexicon = opinion_lexicon.negative();
        self.neg_synonyms_lexicon = []
        self.pos_synonyms_lexicon = []
        self.enricher = TW.TweetEnricher()

    def predict_opinion(self, text):
        pos_count, neg_count = self.enricher.sentiment(text)
        neg_count += 1
        pos_count += 1
        if neg_count/pos_count > self.RATIO:
            print("Oppose")
            return -1
        elif pos_count/neg_count > self.RATIO:
            print("Support")
            return 1
        else:
            print("Neutral")
            return 0

    def get_synonyms(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())

        return set(synonyms)

    def predictOpinionAbandoned(self, text):
        """
        Function that predicts whether the given text has a (positive, negative, neutral) opinion
        on the targets.
        :param text: Text that possibly has an opinion on the given targets
        :param targets: The targets on which the opinion has been expressed
        :return: 1: Positive. 0: Neutral. -1: Negative
        """
        tokenizer = treebank.TreebankWordTokenizer()
        pos_words = 1
        neg_words = 1
        tokenized_sent = [word.lower() for word in tokenizer.tokenize(text)]

        y = []

        for word in tokenized_sent:
            if word in self.pos_lexicon:
                pos_words += 1
                y.append(1)  # positive
            elif word in self.neg_lexicon:
                neg_words += 1
                y.append(-1)  # negative
            else:
                y.append(0)  # neutra

        if pos_words/neg_words > self.RATIO:
            print("Support.")
            return 1
        elif neg_words/pos_words > self.RATIO:
            print("Oppose.")
            return -1
        else:
            print("Neutral")
            return 0


#print(testOC.predict_opinion("What Camaron is completely untrue, stop this lie!"))
#print(testOC.predict_opinion("I agree with it."))
