import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import cluster
import time

# Use an unsupervised model to create clusters
# extract the center of those clusters
# feed those centers to a decision tree
# extract the rules of said decision tree

df = pd.read_csv('../CSV/PRECLUSTERINGComplete.csv', sep=',', header=0, engine='python',  skipfooter=0)

# skip first columns not containing relevant features
X = df[df.columns[6:]]
Y = df[df.columns[2]]

# create the tree model and the tree itself
DTC = DecisionTreeClassifier(random_state=None, max_depth=5)
DTC = DTC.fit(X, Y)

print('Least important feature', df.columns[5+DTC.feature_importances_.argmin()],
      '\nMost important feature', df.columns[5+DTC.feature_importances_.argmax()])

# create the dot file that represents the tree, tune max_depth to change the height of the tree
export_graphviz(DTC, out_file='../CSV/tree.dot', feature_names=df.columns.values[6:], max_depth=2, filled=True, impurity=False)




