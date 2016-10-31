import math
from textblob import TextBlob as tb

class RumorExtractor:
    """
    Class that realises the Rumor Extractor component.
    """

    def __init__(self):
        """
        Method that initialises the Rumor Extractor component.
        """
        print("Initialize")

    def tf(self, term, doc):
        """ TODO: adjust formula to wiki
        Method that computes the Term Frequency.
        :param  term: the term or word.
        :param  doc: The document or tweet containing text.
        :return the number of times that a term occurs in document.
        """
        return (float)(doc.words.count(term)) / (float)(len(doc.words))

    def n_containing(self, term, doclist):
        """
        Method that computes the number of documents containing a specific word.
        :param  term: the term or word.
        :param  doclist: The collection of documents or tweets.
        :return the number of documents containing a specific term.
        """
        return (float)(sum(1 for blob in doclist if term in blob.words))

    def idf(self, term, doclist):
        """
        Method that computes the Inverse Document Frequency for each term, e.g., how common or rare it is across all documents.
        :param  term: the term or word.
        :param  doclist: The collection of documents or tweets.
        :return the inverse document frequency of a term on a document doc.
        """
        return (float)(math.log(len(doclist)) / (float)((1 + self.n_containing(term, doclist))))

    def tfidf(self, term, doc, doclist):
        """
        Method that computes the TF-IDF.
        :param  term: the term or word.
        :param  doc: The document or tweet containing text.
        :param  doclist: The collection of documents or tweets.
        :return TF(term, doc) * IDF(term, doclist)
        """
        return (float)((float)(self.tf(term, doc)) * (float)(self.idf(term, doclist)))

    def computeSimilarity(self, vector1, vector2):
        """
        Method that computes the similarity between two vectors.
        :param vector1: The first vector to compute the similarity.
        :param vector2: The second vector to compute the similarity.
        :return the dot-product of these vectors divided by the product of the magnitude of these vectors.
        """
        dot_product = sum(vector1[key] * vector2.get(key, 0) for key in vector1)
        magnitude_v1 = math.sqrt(sum(math.pow(vector1[key], 2) for key in vector1))
        magnitude_v2 = math.sqrt(sum(math.pow(vector2[key], 2) for key in vector2))
        return (float)(dot_product) / ((float)(magnitude_v1) * (float)(magnitude_v2))

    def mergeClusters(self, clusters, cluster1, cluster2):
        """
        Method that merges two clusters.
        :param cluster1: The first cluster to merge.
        :param cluster2: The second cluster to merge.
        :return the new set of clusters.
        """
        if(cluster1 != cluster2):
            merge = tb(cluster1.string + cluster2.string)
            clusters.remove(cluster1)
            clusters.remove(cluster2)
            clusters.append(merge)
        return clusters