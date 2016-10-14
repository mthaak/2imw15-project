import sklearn
import numpy
from sklearn.model_selection import cross_val_score

class UserClassifier:

    def __init__(self, classifier):
        assert hasattr(classifier, 'fit') and callable(classifier.fit)
        assert hasattr(classifier, 'predict') and callable(classifier.predict)
        self.classifier = classifier
        self.isTrained = False

    def train(self, X, y):
        assert X.shape[0] == len(y)
        self.classifier.fit(X, y)
        self.isTrained = True

    def classify(self, X):
        assert self.isTrained
        return self.classifier.predict(X)

    def evaluate(self, X, y, n_folds = 5):
        f1_scores = cross_val_score(self.classifier, X, y, scoring='f1_weighted', cv=n_folds)
        return numpy.mean(f1_scores)
