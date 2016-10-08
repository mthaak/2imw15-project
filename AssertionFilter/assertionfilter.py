import sklearn
import numpy


class AssertionFilter:
    """
    Class that realizes the Assertion Filter component.
    """
    classifier = None
    is_trained = False
    keep_label = 1  # label denoting assertion

    def __init__(self, classifier):
        """
        Initializes an AssertionFilter with given classifier.
        :param classifier: object implementing fit and predict methods
        """
        assert hasattr(classifier, 'fit') and callable(classifier.fit)
        assert hasattr(classifier, 'predict') and callable(classifier.predict)
        self.classifier = classifier

    def train(self, X, y):
        """
        Trains the classifier.
        :param X: matrix features
        :param y: array of labels
        """
        assert X.shape[0] == len(y)
        self.classifier.fit(X, y)
        self.is_trained = True

    def classify(self, X):
        """
        Returns the predicted labels for a matrix of tweet features.
        :param X: matrix of features
        :return: array of labels
        """
        assert self.is_trained
        return self.classifier.predict(X)

    def filter(self, tweets, X, y):
        """
        Filters the tweets based on their label.
        :param tweets: array of tweets
        :param X: matrix of features
        :param y: array of labels
        :return: array of filtered tweets, matrix of features and array of labels
        """
        assert len(tweets) == X.shape[0] == len(y)
        filter_ids = [i for i in range(len(y)) if y[i] == self.keep_label]
        return tweets[filter_ids], X[filter_ids], y[filter_ids]
