from UserAnalyzer.DataProcessor import *
from UserAnalyzer.UserClassifier import *

from sklearn import tree
import numpy as np

testDP = DataProcessor("./Data/tweets.csv")
testDP.process([])
testCF = UserClassifier(tree.DecisionTreeRegressor())

features = np.array(testDP.features)
labels = np.array(testDP.labels)
print(testCF.evaluate(features, labels))

###########
CFs = {}
for topic in testDP.getTopics():
    X, y = testDP.outPutData(topic)
    cf = CFs[topic] = UserClassifier(tree.DecisionTreeRegressor())
    print(cf.evaluate(X,y))
