import sklearn

class OpinionClassifier:

    def predictOpinion(self, text, targets):
        """
        Function that predicts whether the given text has a (positive, negative, neutral) opinion
        on the targets.
        :param text: Text that possibly has an opinion on the given targets
        :param targets: The targets on which the opinion has been expressed
        :return: 1: Positive. 0: Neutral. -1: Negative
        """

        return 0