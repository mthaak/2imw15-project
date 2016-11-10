import csv
from pylab import *

"""
    Enropies of features added is calculated and compared.
"""

READ_FILENAME="../Data/tweets_ 20161024_111847_assertionlabeled.csv"

labels= {}

def calculate_feature_entropy(feature_name, labels, initial_entropy):
    count_1 = 0
    count_0 = 0
    count_1_1 = 0.0000001
    count_1_0 = 0.0000001
    count_0_1 = 0.0000001
    count_0_0 = 0.0000001

    for i in range(1000):
        if feature_name[i, 0] == 1:
            if labels.get(i) == '1':
                count_1_1 += 1
            else:
                count_1_0 += 1
            count_1 += 1
        elif feature_name[i, 0] == 0:
            if labels.get(i) == '1':
                count_0_1 += 1
            else:
                count_0_0 += 1
            count_0 += 1

    if count_0 == 0 and count_1 == 0:
        entropy = math.inf
    elif count_1 == 0:
        entropy = count_0 / 1000* (count_0_1 / count_0 * math.log(count_0_1 / count_0) + count_0_0 / count_0 * math.log(count_0_0 / count_0))
    elif count_0 == 0:
        entropy = count_1 / 1000* (count_1_1 / count_1 * math.log(count_1_1 / count_1) + count_1_0 / count_1 * math.log(count_1_0 / count_1))
    else:
        entropy = count_1 / 1000 * (
        count_1_1 / count_1 * math.log(count_1_1 / count_1) + count_1_0 / count_1 * math.log(count_1_0 / count_1)) + \
                  count_0 / 1000 * (
                  count_0_1 / count_0 * math.log(count_0_1 / count_0) + count_0_0 / count_0 * math.log(
                      count_0_0 / count_0))

    return -1 * entropy, initial_entropy - (-1 * entropy)




#Read tweet data from file
with open(READ_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):
        #build document for n-grmas count
        if i > 0:
            labels[i-1] = row[12]
        if i == 1000:
            break


count =0
count_1 = 0

label_1 = {}
label_0= {}

# entropy of labelled data
for label in labels:
    label_1[label] = 0
    label_1[label] = 0
    if labels.get(label) == '1':
        count_1+=1
        label_1[label] = 1
    if labels.get(label)  == '0':
        count+=1
        label_0[label] = 0

initial_entropy = count_1/1000 * math.log(count_1/1000) + count/1000 * math.log(count/1000)
initial_entropy = -1 * initial_entropy
print(initial_entropy)

feature = pickle.load(open('../Data/Features_binary.pickle', 'rb'))
print(feature.get_shape())
features = {}
order = {
    0 : 'Vulgar Words',
    1: 'Emoticons',
    2: 'Interrogative',
    3: 'Exclamatory' ,
    4: 'Abbreviations',
    5: 'TwitterJargons' ,
    6: '# presence',
    7: '# position',
    8: '@ presence',
    9: '@position',
    10: 'RT presence',
    11: 'RT position' ,
    12: 'URL presence',
    276: 'ManyNumbers' ,
    277: 'Non-ascii characters' ,
    278: 'LinksToTrusted' ,
    279: 'NegativeOpnions',
    280: 'PositiveOpnions'}
for i in range(13):
    features[i] = (feature.getcol(i), order.get(i))

for i in range(276, 281):
    features[i] = (feature.getcol(i), order.get(i))

gain = []

for i in features:
    entropy, information_gain = calculate_feature_entropy(features.get(i)[0], labels, initial_entropy)
    gain.append((features.get(i)[1],information_gain))
    print(features.get(i)[1])
    print(entropy)
    print(information_gain)

val = []
pos= []
count = 1
for i in gain:
    count += 1
    val.append(i[1])    # the bar lengths
    pos.append(count)    # the bar centers on the y axis

figure(1)
barh(pos,val, align='center')
yticks(pos, [i[0] for i in gain])
xlabel('Information Gain')
title('Feature Evaluation')
grid(False)

show()