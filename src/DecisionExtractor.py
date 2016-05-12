import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import cluster
import time

# Use an unsupervised model to create clusters
# extract the center of those clusters
# feed those centers to a decision tree
# extract the rules of said decision tree

df = pd.read_csv('../CSV/parsedstats23.csv', sep=',', header=0, engine='python',  skipfooter=0)

# skip sha, name, certificate and package
data = df[df.columns[4:]].values

model = cluster.KMeans(n_clusters=400, random_state=1)
start = time.time()
# fit the date and compute compute the clusters
predicted = model.fit_predict(data)
# transform the model to a distance based model
transformed = model.transform(data)
end = time.time()
print('Model execution time ', end-start)

# extract the centers
centers = model.cluster_centers_

# compute the 2 closest nodes to all cluster centers
centroids = []
for j in range(0, max(predicted)):
    ind = np.argsort(transformed[:, j])[::][:2]
    centroids.extend(ind)

# insert the cluster column in the dataframe in order to ease future operations
df.insert(4, 'cluster', predicted)

# extract the features and target labels to classify
output = df.iloc[centroids]
X = output[output.columns[5:]]
Y = output[output.columns[4]]

# create the tree model and the tree itself
DTC = DecisionTreeClassifier(random_state=1, min_samples_leaf=2, max_depth=20)
DTC = DTC.fit(X, Y)
# create the dot file that represents the tree
export_graphviz(DTC, out_file='../CSV/tree.dot', feature_names=df.columns.values[5:])




