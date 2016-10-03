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

    def evaluate(self, X, y, n_folds=10):
        """
        Evaluates the classifier by cross validation and returns the average f1 score.
        :param X: matrix of features
        :param y: array of labels
        :param n_folds: number of folds for cross validation
        :return: float average f1 score
        """
        assert X.shape[0] == len(y)
        assert type(n_folds) == int
        # Calculates the F1-score by cross validation of model
        f1_scores = sklearn.cross_validation.cross_val_score(self.classifier, X, y, scoring='f1_weighted', cv=n_folds)
        return numpy.mean(f1_scores)
