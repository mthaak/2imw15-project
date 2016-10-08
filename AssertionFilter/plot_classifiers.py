import matplotlib.pyplot as plt
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D

"""
    This script reads the results from evaluate_classifiers.py and plots the average F1-score with respect to two
    parameters in a trisurf plot. It also finds the best result.
"""

READ_FILENAME = "./evaluate_classifiers_results.csv"
X_AXIS_COLUMN = 2  # 1st parameter
Y_AXIS_COLUMN = 3  # 2nd parameter
Z_AXIS_COLUMN = 0  # Avg. F1-score

header = None
X, Y, Z = [], [], []
best_result = [0.0]
with open(READ_FILENAME, encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for i, row in enumerate(reader):
        # Get header row
        if not header:
            header = row
            continue
        # Determine x, y and score
        x, y, score = float(row[X_AXIS_COLUMN]), float(row[Y_AXIS_COLUMN]), float(row[Z_AXIS_COLUMN])
        X.append(x)
        Y.append(y)
        Z.append(score)
        # Check if best result
        if score > float(best_result[Z_AXIS_COLUMN]):
            best_result = row

# Not really needed in our case
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

print("Best:", dict(zip(header, best_result)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(X, Y, Z, cmap=plt.cm.jet)
fig.suptitle(best_result[1], fontsize=20)  # assuming classifier name is on the 2nd row
ax.set_xlabel(header[X_AXIS_COLUMN])
ax.set_ylabel(header[Y_AXIS_COLUMN])
ax.set_zlabel("Avg. F1-Score")

plt.show()
